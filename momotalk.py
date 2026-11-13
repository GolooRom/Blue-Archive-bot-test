from itertools import takewhile
import pywikibot as pwb
from pywikibot import Page
from pywikibot.pagegenerators import PreloadingGenerator
import sys

from utils import get_character_table, load_momotalk, load_favor_schedule, save_page

sys.stdout.reconfigure(encoding='utf-8')


def find_convergence_point(conversations: list[dict], options: list[dict]) -> tuple[int, list[list[int]]]:
    """
    When dialogue branches based on Sensei's choice, find out at which point the conversation
    converges to the same lines.

    Args:
        conversations (list[dict]): all lines in this conversation
        options (list[int]): a list of all possible options

    Returns:
        tuple[int, list[list[int]]]: convergence point and ids of unique lines for each options
    """
    conversation_dict: dict[int, list[dict]] = {}
    for c in conversations:
        gid = c['MessageGroupId']
        if gid not in conversation_dict:
            conversation_dict[gid] = []
        conversation_dict[gid].append(c)

    paths: list[list[int]] = []
    for option_index, option in enumerate(options):
        paths.append([])
        oid = option['NextGroupId']
        paths[option_index].append(oid)
        while oid in conversation_dict:
            next_id = conversation_dict[oid][-1]['NextGroupId']
            if next_id not in conversation_dict:
                break
            paths[option_index].append(next_id)
            oid = next_id
    path_sets: list[set[int]] = []
    for path in paths:
        path_set = set()
        for cid in path:
            path_set.add(cid)
        path_sets.append(path_set)
    convergence = -1
    for cid in paths[0]:
        for path_set in path_sets:
            if cid not in path_set:
                break
        else:
            convergence = cid
            break
    # assert convergence != -1, str(options)

    return convergence, [list(takewhile(lambda x: x != convergence, path)) for path in paths]


student_reply_group: int = 0


def make_conversation(conversation: list[dict], char_name: str,
                      conversation_counter: int,
                      unlock_favor: int) -> str:
    result = [f'<span id="momotalk-{conversation_counter}"></span><span id="momotalk-favor-{unlock_favor}"></span>',
              "{{MomoTalk"]
    char_name_short = char_name.split()[0]
    n = len(conversation)

    prev_group_id = -1
    i = 0
    counter = 1
    global student_reply_group
    relationship_event_found = 0
    reply_convergence = -1
    reply_group: dict[int, int] = {}
    no_favor_schedule: set[int] = set()
    while i < n:
        c = conversation[i]
        group_id = c['MessageGroupId']
        if c['MessageCondition'] == "Answer":
            # Sensei's line
            if i < n - 1 and group_id == conversation[i + 1]['MessageGroupId']:
                candidates = [c]
                while i < n - 1 and conversation[i]['MessageGroupId'] == conversation[i + 1]['MessageGroupId']:
                    i += 1
                    candidates.append(conversation[i])
                options = ''.join(
                    f'|option{counter}_{index + 1}={c["MessageEN"]}\n' for index, c in enumerate(candidates))
                line = f"|{counter}=reply\n{options}"
                reply_convergence, reply_list = find_convergence_point(conversation, candidates)
                for index, replies in enumerate(reply_list):
                    if len(replies) > 0:
                        no_favor_schedule.add(replies[-1])
                    for reply_id in replies:
                        reply_group[reply_id] = index + 1
                if len(reply_list) > 0 and len(reply_list[-1]) > 0:
                    no_favor_schedule.remove(reply_list[-1][-1])
                student_reply_group += 1
                line += f"|group{counter}={student_reply_group}\n"
            else:
                line = f"|{counter}=sensei\n|text{counter}={c['MessageEN']}\n"
                if group_id in reply_group:
                    line += f"|group{counter}={student_reply_group}\n|option{counter}={reply_group[group_id]}\n"
        else:
            # student response
            if c['MessageType'] == 'Image':
                line = f"|{counter}=student-image\n|file{counter}={c['ImagePath'].split('/')[-1]}"
            else:
                message = c['MessageEN']
                if message.strip() == "":
                    message = "&nbsp;"
                line = f"|{counter}=student-text\n|text{counter}={message}"
            if group_id != prev_group_id:
                line += f"\n|name{counter}={char_name_short}\n|profile{counter}={char_name}"
            line += "\n"
            if group_id in reply_group:
                line += f"|group{counter}={student_reply_group}\n|option{counter}={reply_group[group_id]}\n"
        # extra conditions ensure that relationship popup does not appear twice
        # Izumi has a false positive where is FavorScheduleId is non-zero but an event shouldn't appear.
        #  This is ruled out by the last check.
        if c['FavorScheduleId'] != 0 and group_id not in no_favor_schedule and c['FavorScheduleId'] != c['PreConditionFavorScheduleId']:
            counter += 1
            line += f"\n|{counter}=relationship\n|name{counter}={char_name}\n"
            line += f"|favor{counter}={unlock_favor}\n|sequence{counter}={conversation_counter}\n"
            relationship_event_found += 1
            assert relationship_event_found == 1, f"For {char_name}: relationship event should occur exactly once; occurred {relationship_event_found} time(s) instead. Last group id: {group_id}"
        result.append(line)
        prev_group_id = group_id
        i += 1
        counter += 1

    result.append("}}")

    return "\n".join(result)


def make_top(student_name: str):
    return f"""{{{{MomoTalkTop}}}}{{{{#seo:
    |title={student_name} MomoTalks
    |title_mode=append
    |keywords={student_name},MomoTalk,chat,messenger,ingame,dialogue,Blue Archive,モモトーク
    |description=All MomoTalk chats with {student_name}
    |image = {student_name}.png
    |image_alt = {student_name} MomoTalks
}}}}"""


def make_character_momotalk(momotalk: list[dict], char_name: str, favor_levels: list[int]) -> str:
    suspicion = 0
    for t in momotalk:
        if t['MessageEN'] == "" and t['MessageType'] != "Image":
            suspicion += 1
    if suspicion > 3:
        print(f"ERROR: {char_name}'s MomoTalk contains too much whitespace")
        return ""
    if char_name.startswith("Arisu"):
        char_name = char_name.replace("Arisu", "Aris")

    current = []
    result = []
    counter = 1
    global student_reply_group
    student_reply_group = 0

    def add_conversation():
        result.append(make_conversation(current, char_name, counter, favor_levels[len(result)]))

    for m in momotalk:
        if m['MessageCondition'] == 'FavorRankUp':
            if len(current) > 0:
                add_conversation()
                counter += 1
                current = []
        current.append(m)
    add_conversation()

    result.append("[[Category:MomoTalk]]")
    return make_top(char_name) + "\n\n".join(result)


s = pwb.Site()


confirm = False


def create_momotalk_page(p: pwb.Page, student_name, text):
    global confirm
    if text.strip() == "":
        print(f"ERROR: text for {student_name} is empty.")
        return
    setattr(p, "_bot_may_edit", True)
    if text == p.text:
        # print(f"INFO: {student_name} has the same text.")
        return
    if confirm:
        x = input(f"Save {p.title()}? ")
        if x.lower() == 'a':
            confirm = False
    save_page(p, text, summary="auto-generate momotalk")


def get_character_favor_schedule(char_id: int) -> list[int]:
    events = load_favor_schedule()[char_id]
    return list(sorted(e['FavorRank'] for e in events))


def momotalk_main():
    char_dict = get_character_table(use_cache=False)
    momotalk_dict = load_momotalk()
    results: list[tuple[str, str]] = []
    for char_id, momotalk in momotalk_dict.items():
        if char_id not in char_dict:
            continue
        try:
            favor_schedule = get_character_favor_schedule(char_id)
        except Exception as e:
            print(char_id)
            continue
        char_name = char_dict[char_id]
        char_name = char_name[0].capitalize() + char_name[1:]
        momotalk_text = make_character_momotalk(momotalk, char_name, favor_schedule)
        results.append((char_name, momotalk_text))
    pages = (pwb.Page(s, f"{pair[0]}/MomoTalk") for pair in results)
    pages = list(PreloadingGenerator(pages))
    for index, p in enumerate(pages):
        create_momotalk_page(p, results[index][0], results[index][1])
    print("MomoTalk done")



if __name__ == "__main__":
    momotalk_main()
