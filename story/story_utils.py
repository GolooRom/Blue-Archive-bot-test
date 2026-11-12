import json
import re
from dataclasses import dataclass, asdict, field
from enum import Enum
from functools import cache, cached_property
from pathlib import Path

from pywikibot import Page
from pywikibot.pagegenerators import GeneratorFactory
from wikitextparser import Template, parse

from story.log_utils import logger
from utils import scenario_character_name, dev_name_to_canonical_name, load_json, load_json_list, s, find_template, \
    save_page

@cache
def get_existing_sprites() -> dict[str, list[str]]:
    result_file = Path("cache/sprites.json")
    if not result_file.exists():
        gen = GeneratorFactory(s)
        gen.handle_args(['-cat:Character sprites', '-cat:Character sprite redirects', '-ns:File'])
        gen = gen.getCombinedGenerator()
        result: dict[str, list[str]] = {}
        for p in gen:
            title = p.title(with_ns=False, underscore=False)
            name, num = re.search(r"^(.*)[ _](\d\d)\.png", title).groups()
            if name not in result:
                result[name] = []
            result[name].append(num)
        json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return json.load(open(result_file, "r", encoding="utf-8"))

reported_missing_spines: set[str] = set()


def run_hash(string):
    from xxhash import xxh32
    return int(xxh32(string).intdigest())

def get_scenario_character_id(text_ko_original: str) -> tuple[list[tuple[str, str, str, str]], int]:
    result = []
    speaker = None
    if len(scenario_character_name) == 0:
        path = Path("json/ScenarioCharacterNameExcelTable.json")
        loaded = json.load(open(path, "r", encoding="utf-8"))
        loaded = loaded['DataList']
        for row in loaded:
            cid = row['CharacterName']
            scenario_character_name[cid] = row
    # search_text = re.search(r"^\d+;([^;a-zA-Z]+) ?([a-zA-Z]+)?;\d+;?", text_ko)
    for original in text_ko_original.split("\n"):
        # deal with cases such as 3;사키;S2_11
        search_text = re.search(r"^\d+;([^;]+);([S_\d]+);?", original)
        if search_text is not None:
            name_ko = search_text.group(1)
            expression_number = search_text.group(2)
            na = False
        else:
            search_text = re.search(r"#na;([^\n#;]+)(;.+)?", original)
            if search_text is not None:
                na = True
                if search_text.group(2) is not None:
                    name_ko = search_text.group(1)
                else:
                    result.append([None, None, None, None, None])
                    continue
                expression_number = None
            else:
                continue

        # a -> A; b -> B
        string_options = [name_ko, name_ko.upper(), name_ko.lower(), name_ko.encode('utf-8')]
        for string in string_options:
            hashed = run_hash(string)
            if hashed in scenario_character_name:
                break
        else:
            logger.warning(f"Cannot find scenario character name in table. Text: {name_ko}. Hash: {hashed}.")
            continue
        row = scenario_character_name[hashed]
        name = row['NameEN']
        nickname = row['NicknameEN']
        spine = row['SpinePrefabName'].split("/")[-1]
        if spine is not None and spine.strip() != "":
            spine = spine.replace("CharacterSpine_", "")
            spine = dev_name_to_canonical_name(spine)
        portrait = row['SmallPortrait'].split("/")[-1]
        if portrait is not None and portrait != "":
            if "Student_Portrait_" in portrait:
                portrait = portrait.replace("Student_Portrait_", "")
                portrait = dev_name_to_canonical_name(portrait)
            elif "NPC_Portrait_" in portrait:
                portrait = portrait.replace("NPC_Portrait_", "")
                portrait = dev_name_to_canonical_name(portrait)
        if na:
            spine, portrait = '', ''

        # Sometimes spine does not agree with portrait. In that case, use spine unless spine is empty but portrait
        # is not.
        if spine != portrait and spine.strip() == '' and portrait != '':
            spine = portrait
        existing_sprites = get_existing_sprites()
        if spine != "" and spine not in existing_sprites:
            if (spine + " diorama") in existing_sprites:
                spine += " diorama"
            else:
                if spine not in reported_missing_spines:
                    logger.warning(f"Spine {spine} not found")
                    reported_missing_spines.add(spine)

        # deal with cases such as 3;사키;S2_11 and 3;히마리;S2
        if expression_number is not None and "S" in expression_number:
            match = re.search(r"(S\d?)_(\d\d)", expression_number)
            if match is not None:
                spine_suffix, expression_number = match.groups()
            else:
                match = re.search(r"S\d+", expression_number)
                if match is not None:
                    spine_suffix = match.group(0)
                    expression_number = "00"
                else:
                    raise RuntimeError(f"Spine {expression_number} cannot be parsed")
            spine += " " + spine_suffix

        if spine in existing_sprites and expression_number not in existing_sprites[spine]:
            repl = existing_sprites[spine][0]
            logger.debug(f"{spine}_{expression_number} does not exist. Replacing with {repl}.")
            expression_number = repl

        obj = (name, nickname, spine, portrait, expression_number)
        # check if there is text after the last semicolon; if so, this is the speaker
        if re.search(r"^\d+;([^;]+);([S_\d]+);.", original) is not None or na:
            speaker = obj
        result.append(obj)
    return result, speaker


def get_main_scenarios() -> list[dict]:
    with open ("json/ScenarioModeExcelTable.json", "r", encoding="utf-8") as f:
        result = json.load(f)
        result = result['DataList']
        # Series1's Prologue rows duplicate its Main rows in volume 0, so only Series2
        # contributes ModeType == "Prologue" episodes of its own.
        return [row for row in result
                if row['ModeType'] in {"Main", "SpecialOperation"}
                or (row['ModeType'] == "Prologue" and row['SubType'] == "Series2")]


class StoryType(Enum):
    RELATIONSHIP = 0
    MAIN = 1
    SIDE = 2
    GROUP = 3
    EVENT = 4


def get_story_title_and_summary(query: int, story_type: StoryType) -> tuple[str, str]:
    def process(loaded) -> dict[int, str]:
        data_list = loaded['DataList']
        result: dict[int, str] = {}
        for row in data_list:
            result[row['Key']] = row['En']
        return result

    data = load_json("LocalizeExcelTable.json", process)
    title_key = f"ScenarioDigest_Title_{query}"
    description_key = f"ScenarioDigest_Description_{query}"
    return data.get(run_hash(title_key), None), data.get(run_hash(description_key), None)


def make_nav_span(event: dict) -> str:
    return f'<span id="relationship-{event["OrderInGroup"]}"></span><span id="relationship-favor-{event["FavorRank"]}"></span>'


def em_map(regex: re.Match[str]):
    # mapper = {
    #     '땀': 'sweat',
    #     '속상함': 'anxious',
    #     '반짝': 'twinkle',
    #     '…': 'ellipsis',
    # }
    d = regex.groupdict()
    result = d['m1'] if d["m1"] is not None else d['m2']
    return "{{emoticon|" + result + "}}"


def extract_em(string) -> list[str]:
    regex = re.compile(r"#\d;em;((?P<m1>…)|\[(?P<m2>[^\]]+)\])")
    result = []
    while True:
        match = re.search(regex, string)
        if match is None:
            break
        result.append(em_map(match))
        string = re.sub(regex, "", string)
    return result


def strip_st_line(line: str) -> str:
    line, _ = re.subn(r"\[log=[^]]+]", "", line)
    line = line.replace("[/log]", "")
    line, _ = re.subn(r"\[wa:\d+]", "", line)
    return line


def make_categories(start: list[str] = None, character_list: dict[str, int] = None, bgm_list: set[str] = None) -> str:
    if character_list is None:
        character_list = []
    if bgm_list is None:
        bgm_list = dict()
    if start is None:
        start = set()
    bgm_string = "{{Story/BGMList | " + " | ".join(str(bgm) for bgm in sorted(bgm_list)) + " }}"
    char_string = "{{Story/CharList | " + " | ".join(sorted(character_list.keys())) + " }}"
    return bgm_string + char_string + "\n".join(f"[[Category:{c}]]" for c in start)


def get_events(pattern: str) -> dict:
    def process(d: list[dict]) -> dict[int, list[dict]]:
        result: dict[int, list[dict]] = {}
        for loaded in d:
            loaded = loaded['DataList']
            for row in loaded:
                teen_mode = row['TeenMode']
                if teen_mode:
                    continue
                group_id = row['GroupId']
                if group_id not in result:
                    result[group_id] = []
                result[group_id].append(row)
        return result

    file_names = tuple(pattern.format(i) for i in range(1, 10))
    e = load_json_list(file_names, process)
    return e


def get_story_event(query_group_id: int) -> list[dict] | None:
    e = get_events("ScenarioScriptExcelTable{0}.json")
    return e.get(query_group_id, None)


@dataclass
class NavArgs:
    prev_title: str = ""
    prev_page: str = ""
    next_title: str = ""
    next_page: str = ""


@dataclass
class StoryInfo:
    title: str
    summary: str
    main_text: str
    category: str
    chars: dict[str, int]
    nav_top: Template = field(default_factory=lambda: Template("{{Story/StoryTop}}"))
    nav_bottom: Template = field(default_factory=lambda: Template("{{Story/StoryBottom}}"))
    # main story pages write their nav templates without spaces around the pipes
    nav_compact: bool = False

    # FIXME: the before argument should be dropped
    def add_nav_arg(self, k: str, v: str, top_only: bool = False, before: str = None):
        self.nav_top.set_arg(k, v, before=before)
        if not top_only:
            self.nav_bottom.set_arg(k, v)

    @property
    def text(self):
        raise RuntimeError()

    @cached_property
    def full_text(self):
        def format_nav(nav: str) -> str:
            if self.nav_compact:
                return nav
            nav, _ = re.subn(r"(?<! )\|", " |", nav)
            nav, _ = re.subn(r"\|(?! )", "| ", nav)
            nav, _ = re.subn(r"(?<! )}}", " }}", nav)
            return nav

        return "\n".join([
            format_nav(str(self.nav_top)),
            self.main_text,
            format_nav(str(self.nav_bottom)),
            self.category
        ])


def make_story_nav(story: StoryInfo,
                   nav_args: NavArgs):
    args = asdict(nav_args)
    # FIXME: don't fix ordering
    for k in ['next_title', 'next_page', 'prev_title', 'prev_page']:
        v = args[k]
        if v == "":
            continue
        story.add_nav_arg(k, v, before="title")


def migrate_editor_nav_links(nav: Template, existing_text: str):
    """The bot always owns the unnumbered prev/next slot. If the page has a different link there
    and also uses the numbered slots (e.g. next_page2), an editor put it there, so move it to a
    numbered slot: the slot that holds the generated link if the editor moved it there,
    otherwise the first free slot from 2. Without next_page2, the link is just an outdated bot
    link and is overwritten."""
    existing = find_template(parse(existing_text), nav.name.strip())
    if existing is None:
        return

    def normalize(title: str) -> str:
        title = title.replace("_", " ").strip()
        return title[:1].upper() + title[1:]

    def value(key: str) -> str | None:
        arg = existing.get_arg(key)
        return None if arg is None else arg.value.strip()

    for direction in ["prev", "next"]:
        page_key, title_key = f"{direction}_page", f"{direction}_title"
        generated_page = nav.get_arg(page_key)
        editor_page = value(page_key)
        if generated_page is None or editor_page is None:
            continue
        if normalize(editor_page) == normalize(generated_page.value):
            continue
        if existing.get_arg(f"{page_key}2") is None:
            continue
        suffixes = set()
        target = None
        for arg in existing.arguments:
            match = re.fullmatch(rf"{direction}_(?:page|title)(\d+)", arg.name.strip())
            if match is None:
                continue
            suffixes.add(int(match.group(1)))
            if (target is None and arg.name.strip().startswith(page_key)
                    and normalize(arg.value) == normalize(generated_page.value)):
                target = int(match.group(1))
        if target is None:
            target = 2
            while target in suffixes:
                target += 1
        nav.set_arg(f"{page_key}{target}", editor_page)
        editor_title = value(title_key)
        if editor_title is not None:
            nav.set_arg(f"{title_key}{target}", editor_title)


def keep_extra_nav_args(nav: Template, existing_text: str):
    """Copy over arguments that editors added to the nav template on the page and that the bot
    does not generate, such as extra buttons (next_page2, prev_title3, ...) linking to exploration
    stories. The bot keeps updating the arguments it generates and leaves the rest alone."""
    existing = find_template(parse(existing_text), nav.name.strip())
    if existing is None:
        return
    generated = {arg.name.strip() for arg in nav.arguments}
    # MediaWiki uses the last value when an argument is repeated
    extra = {arg.name.strip(): arg.value.strip()
             for arg in existing.arguments
             if not arg.positional and arg.name.strip() not in generated}
    for name, value in extra.items():
        nav.set_arg(name, value)


def match_nav_arg_order(nav: Template, existing_text: str):
    """Reorder nav's arguments to follow the order the same template already uses on the page.

    Main story pages predate the current generator and list title/summary before the prev/next
    links; regenerating them would otherwise rewrite every page just to shuffle arguments."""
    existing = find_template(parse(existing_text), nav.name.strip())
    if existing is None:
        return
    args = nav.arguments
    if any(arg.positional for arg in args):
        return
    values = {arg.name.strip(): arg.value for arg in args}
    if len(values) != len(args):
        # duplicated argument names; deleting by name would lose data
        return
    order = [arg.name.strip() for arg in existing.arguments]
    # stable sort, so arguments absent from the existing page keep their order at the end
    names = sorted(values, key=lambda name: order.index(name) if name in order else len(order))
    if names == list(values):
        return
    for name in names:
        nav.del_arg(name)
    for name in names:
        nav.set_arg(name, values[name])


def nav_is_compact(nav: Template, existing_text: str) -> bool | None:
    """Whether the page writes this nav template without spaces around its pipes. None if the
    page has no such template to copy from."""
    existing = find_template(parse(existing_text), nav.name.strip())
    if existing is None or len(existing.arguments) == 0:
        return None
    return all(not arg.string[1:2].isspace() for arg in existing.arguments)


def save_story_page(page: Page, story: StoryInfo, summary: str):
    existing_text = page.text
    migrate_editor_nav_links(story.nav_top, existing_text)
    migrate_editor_nav_links(story.nav_bottom, existing_text)
    keep_extra_nav_args(story.nav_top, existing_text)
    keep_extra_nav_args(story.nav_bottom, existing_text)
    match_nav_arg_order(story.nav_top, existing_text)
    match_nav_arg_order(story.nav_bottom, existing_text)
    compact = nav_is_compact(story.nav_top, existing_text)
    if compact is not None:
        story.nav_compact = compact
    # full_text is cached, so it must not have been rendered before the nav templates settled
    story.__dict__.pop("full_text", None)
    save_page(page, story.full_text, summary=summary)


def make_story_list_nav(stories: list[StoryInfo], page_prefix: str):
    for index, story in enumerate(stories):
        args = NavArgs()
        if index > 0:
            args.prev_title = stories[index - 1].title
            args.prev_page = f"{page_prefix}{index}"
        if index < len(stories) - 1:
            args.next_title = stories[index + 1].title
            args.next_page = f"{page_prefix}{index + 2}"
        make_story_nav(story, args)