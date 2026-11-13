import dataclasses
import pickle
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pywikibot import Page, Site, showDiff
from pywikibot.pagegenerators import GeneratorFactory
from wikitextparser import parse, WikiText, Template

import json


s = Site()


json_cache: dict[str | tuple[str, ...], Any] = {}


def load_json[T](file_name: str, processor: Callable[[dict], T] = lambda x: x) -> T | None:
    if file_name not in json_cache:
        path = Path("json") / file_name
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            json_cache[file_name] = processor(json.load(f))
    return json_cache[file_name]


def load_json_list[T](files: tuple[str, ...], processor: Callable[[list[dict]], T]) -> T:
    if files not in json_cache:
        json_dicts = []
        for f in files:
            p = Path("json") / f
            if p.exists():
                json_dicts.append(json.load(open(p, "r", encoding="utf-8")))
        json_cache[files] = processor(json_dicts)
    return json_cache[files]


def normalize_char_name(original: str) -> str:
    return re.sub(r" ?\(.+\)", "", original)


def get_character_table(use_cache: bool = True) -> dict[int, str]:
    path = Path("cache/char_id.pickle")
    if path.exists() and use_cache:
        result = pickle.load(open(path, "rb"))
    else:
        path.parent.mkdir(exist_ok=True)
        gen = GeneratorFactory(s)
        gen.handle_args(["-cat:Characters"])
        gen = gen.getCombinedGenerator(preload=True)
        result = {}
        for p in gen:
            char_id = int(re.search(r"Id *= *([0-9]+)", p.text).group(1))
            char_name = re.search(r"\| *(Wiki)?[Nn]ame *= *(?P<name>[^\n]+)", p.text).group("name")
            result[char_id] = char_name
        pickle.dump(result, open(path, "wb"))
    return result


@cache
def get_dev_name_map() -> dict[str, str]:
    dev_name_map: dict[str, str] = {}

    def read_dev_name_map(fname: str):
        name_map = json.load(open(fname, "r", encoding="utf-8"))
        for k, v in name_map.items():
            wiki_name = v['firstname']
            if v['variant'] is not None:
                wiki_name += f" ({v['variant']})"
            dev_name_map[k] = wiki_name

    read_dev_name_map('json/devname_map.json')
    read_dev_name_map('json/devname_map_aux.json')

    return dev_name_map


def dev_name_to_canonical_name(dev_name: str) -> str:
    dev_name_map = get_dev_name_map()

    if dev_name == "Null":
        return ""
    if dev_name in dev_name_map:
        return dev_name_map[dev_name]
    if dev_name.capitalize() in dev_name_map:
        return dev_name_map[dev_name.capitalize()]
    if dev_name.lower() in dev_name_map:
        return dev_name_map[dev_name.lower()]
    # print("Cannot find canonical name of " + dev_name)
    return ""


@cache
def get_club_name_mapping() -> dict[str, str]:
    path = Path("json/glossary.py")
    assert path.exists()
    result: dict[str, str] = {}
    for line in open(path, "r", encoding="utf-8").readlines():
        m = re.search(r"'(.+)': ?'(.+)'(,|$)", line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def get_localized_club_name(name: str):
    mapping = get_club_name_mapping()
    if name in mapping:
        return mapping[name]
    raise ValueError(name)


@cache
def load_momotalk() -> dict[int, list[dict]]:
    result = {}
    momotalk = load_json(f"AcademyMessangerExcelTable.json")
    if 'DataList' in momotalk:
        momotalk = momotalk['DataList']
    for talk in momotalk:
        cid = talk['CharacterId']
        if cid not in result:
            result[cid] = []
        result[cid].append(talk)
    return result


@cache
def load_favor_schedule() -> dict[int, list[dict]]:
    favor_schedule = {}
    loaded = load_json("AcademyFavorScheduleExcelTable.json")
    loaded = loaded['DataList']
    for row in loaded:
        cid = row['CharacterId']
        if cid not in favor_schedule:
            favor_schedule[cid] = []
        favor_schedule[cid].append(row)
    return favor_schedule


scenario_character_name: dict[int, dict] = {}

background_file_name: dict[int, str] = {}


def get_background_file_name(background_id: int) -> str | None:
    if len(background_file_name) == 0:
        loaded = json.load(open("json/ScenarioBGNameExcelTable.json", "r", encoding="utf-8"))
        loaded = loaded['DataList']
        for row in loaded:
            bg_id = row['Name']
            bg_name: str = row['BGFileName']
            background_file_name[bg_id] = bg_name.split("/")[-1]
    return background_file_name.get(background_id, None)


@dataclass
class BGMInfo:
    id: int
    name: str
    loop_start: float
    loop_end: float
    volume: float
    transition_time: float
    offset_time: float


bgm_file_info: dict[int, BGMInfo] = {}


def get_bgm_file_info(query_id: int) -> BGMInfo:
    def list_or_none(l: list) -> str | None:
        if len(l) == 0:
            return None
        return l[0]
    
    if len(bgm_file_info) == 0:
        loaded = json.load(open("json/BGMExcelTable.json", "r", encoding="utf-8"))
        loaded = loaded['DataList']
        for row in loaded:
            bgm_id = row['Id']
            bgm_name: str = list_or_none(row['Path']).split("/")[-1]
            loop_start = list_or_none(row['LoopStartTime'])
            loop_end = list_or_none(row['LoopEndTime'])
            volume = list_or_none(row['Volume'])
            transition_time = list_or_none(row['LoopTranstionTime'])
            offset_time = list_or_none(row['LoopOffsetTime'])
            info_list = [loop_start, loop_end, volume, transition_time, offset_time]
            info_list = [None if x is None else float(x) for x in info_list]
            bgm_file_info[bgm_id] = BGMInfo(bgm_id, bgm_name, *info_list)
    if query_id in bgm_file_info:
        return bgm_file_info[query_id]
    raise RuntimeError(f"Bgm with id {query_id} not found.")


music_dict: dict[int, str] = {}


def get_music_info(music_id: int) -> str:
    if len(music_dict) == 0:
        p = Page(s, "Music")
        parsed = parse(p.text)
        for t in parsed.templates:
            if t.name.strip() != "Track":
                continue
            music_dict[int(t.get_arg("Id").value)] = t.get_arg("Title").value.strip()
    return music_dict.get(music_id, "")


def get_music_dict() -> dict[int, str]:
    get_music_info(0)
    return music_dict


def music_file_name_to_title(file_name: str) -> str:
    bgm_id_match = re.search(r"\d+", file_name)
    if bgm_id_match is None:
        return file_name
    bgm_id = int(bgm_id_match.group(0))
    bgm_name = get_music_info(bgm_id)
    if bgm_name == "":
        return file_name
    return bgm_name


def save_json_page(page: Page | str, obj, summary: str = "update json page"):
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    def dump(o):
        return json.dumps(o, indent=4, cls=EnhancedJSONEncoder)

    if isinstance(page, str):
        page = Page(s, page)

    if get_ignore_rules(page.title())[0]:
        print(f"Ignored {page.title()} (in ignore_list.json)")
        return

    if page.text != "":
        original = dump(json.loads(page.text))
    else:
        original = ""
    modified = dump(obj)
    if original != modified:
        page.text = modified
        page.save(summary=summary)


def signature_escape(original: str) -> str:
    res, _ = re.subn(r"~~(?=~)", "~~&#x200B;", original)
    return res


def find_template(w: WikiText, name: str) -> Template | None:
    for t in w.templates:
        if t.name.strip().replace(" ", "_").lower() == name.replace(" ", '_').lower():
            return t
    return None


IGNORE_LIST_PATH = Path(__file__).parent / "ignore_list.json"


def normalize_template_name(name: str) -> str:
    name = name.strip().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^template ?:", "", name, flags=re.IGNORECASE).strip()
    return name.lower()


def normalize_page_title(title: str) -> str:
    """Accepts a plain page title or a full https://bluearchive.wiki/wiki/... URL.
    Underscores become spaces and a trailing slash is dropped."""
    title = title.strip()
    if title.startswith(("http://", "https://")):
        path = urlparse(title).path
        if "/wiki/" not in path:
            raise ValueError(f"Not a wiki page URL: {title}")
        title = unquote(path.split("/wiki/", 1)[1])
    return title.replace("_", " ").strip().rstrip("/").strip()


def parse_template_names(names, where: str) -> frozenset[str]:
    if not isinstance(names, list) or len(names) == 0 or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{IGNORE_LIST_PATH.name}: {where}: expected a non-empty list of template names")
    return frozenset("*" if n.strip() == "*" else normalize_template_name(n) for n in names)


@dataclass(frozen=True)
class IgnoreRule:
    title: str                                         # the page this block is about, e.g. "Some Event/Story"
    single_page: bool                                  # covers the page itself
    number_range: tuple[int, int] | None               # covers the pages "<title>/<number>" for numbers in this range
    all_subpages: bool                                 # covers every page below the page, at any depth
    templates: frozenset[str] | None                   # None = ignore everything, otherwise only these templates

    def matches(self, title: str) -> bool:
        """Does this block cover this (normalized) page title?"""
        if self.single_page and title == self.title:
            return True
        prefix = self.title + "/"
        if not title.startswith(prefix):
            return False
        if self.all_subpages:
            return True
        if self.number_range is not None:
            child = title[len(prefix):]
            if child.isascii() and child.isdigit():
                return self.number_range[0] <= int(child) <= self.number_range[1]
        return False


IGNORE_BLOCK_KEYS = {"page", "single_page", "range", "all_subpages", "ignore"}


@cache
def load_ignore_list() -> list[IgnoreRule]:
    """Reads ignore_list.json: a list with one block per page.

        {
            "page":         "Some Event/Story",      the page (a title or a full wiki URL, no trailing slash needed)
            "single_page":  true,                    cover the page itself
            "range":        [1, 9],                  cover "Some Event/Story/1" to "Some Event/Story/9"
            "all_subpages": true,                    cover every page below the page, at any depth
            "ignore":       "everything"             what the bot must leave alone: "everything" (the default) or
                                                     a list of template names, e.g. ["Story/StoryTop", "Alpha"]
        }

    Pick any of single_page, range and all_subpages. If you pick none, the block covers the page itself.
    A page can have several blocks. Blocks that only have keys starting with "_" are ignored (comments).
    "*" in a template list means every template."""
    if not IGNORE_LIST_PATH.exists():
        return []
    with open(IGNORE_LIST_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    name = IGNORE_LIST_PATH.name
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a list of blocks, one per page. "
                         f"The older format with \"pages\", \"ranges\" and \"templates\" is no longer supported.")

    rules: list[IgnoreRule] = []
    for block in raw:
        if not isinstance(block, dict):
            raise ValueError(f"{name}: every block must be an object like {{\"page\": ..., \"ignore\": ...}}: {block}")
        if len(block) > 0 and all(k.startswith("_") for k in block):
            continue  # a comment
        unknown = [k for k in block if k not in IGNORE_BLOCK_KEYS and not k.startswith("_")]
        if unknown:
            raise ValueError(f"{name}: unknown key(s) {unknown} in {block}, allowed: {sorted(IGNORE_BLOCK_KEYS)}")
        page = block.get("page")
        if not isinstance(page, str) or normalize_page_title(page) == "":
            raise ValueError(f"{name}: every block needs a \"page\": {block}")
        where = f"page {page}"
        title = normalize_page_title(page)

        flags = {}
        for key in ("single_page", "all_subpages"):
            value = block.get(key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name}: {where}: \"{key}\" must be true or false")
            flags[key] = value

        number_range = None
        if "range" in block:
            r = block["range"]
            if (not isinstance(r, list) or len(r) != 2
                    or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in r)
                    or r[0] > r[1]):
                raise ValueError(f"{name}: {where}: \"range\" must be two numbers, the first not after the "
                                 f"second, like [1, 9]")
            number_range = (r[0], r[1])

        if flags["single_page"] is None and flags["all_subpages"] is None and number_range is None:
            flags["single_page"] = True
        if not flags["single_page"] and not flags["all_subpages"] and number_range is None:
            raise ValueError(f"{name}: {where}: this block covers nothing (single_page, range and all_subpages "
                             f"are all off)")

        ignore = block.get("ignore", "everything")
        if isinstance(ignore, str):
            if ignore.strip().lower() != "everything":
                raise ValueError(f"{name}: {where}: \"ignore\" must be \"everything\" or a list of template names")
            templates = None
        else:
            templates = parse_template_names(ignore, where)
        rules.append(IgnoreRule(title, bool(flags["single_page"]), number_range, bool(flags["all_subpages"]),
                                templates))
    return rules


def get_ignore_rules(page_title: str) -> tuple[bool, frozenset[str]]:
    """(ignore the whole page?, names of templates to keep from the wiki) for a page title.
    Ignoring everything wins over keeping templates; template names from several blocks are combined."""
    title = normalize_page_title(page_title)
    templates: set[str] = set()
    for rule in load_ignore_list():
        if rule.matches(title):
            if rule.templates is None:
                return True, frozenset()
            templates |= rule.templates
    return False, frozenset(templates)


def split_into_templates(text: str) -> list[str | Template]:
    """Split wikitext into an ordered list of plain-text chunks and top-level templates.
    Concatenating str() of every element gives back the original text."""
    parsed = parse(text)
    spans = sorted((t.span, t) for t in parsed.templates)
    parts: list[str | Template] = []
    pos = 0
    for (start, end), template in spans:
        if start < pos:
            # nested inside a template that was already taken
            continue
        if start > pos:
            parts.append(text[pos:start])
        parts.append(template)
        pos = end
    if pos < len(text):
        parts.append(text[pos:])
    return parts


def keep_ignored_templates(original_text: str, new_text: str, ignored: frozenset[str]) -> str:
    """Replace every template of new_text that is in `ignored` ("*" = all templates) with the
    template of the same name in original_text. The n-th occurrence of a template in the new
    text takes the n-th occurrence in the original, whether or not they differ. If the original
    has no n-th occurrence (for example a brand new page), the new template is kept."""
    if len(ignored) == 0:
        return new_text

    def is_ignored(template: Template) -> bool:
        return "*" in ignored or normalize_template_name(template.name) in ignored

    original_by_name: dict[str, list[str]] = {}
    for part in split_into_templates(original_text):
        if isinstance(part, Template) and is_ignored(part):
            original_by_name.setdefault(normalize_template_name(part.name), []).append(str(part))

    used: dict[str, int] = {}
    result: list[str] = []
    for part in split_into_templates(new_text):
        if isinstance(part, Template) and is_ignored(part):
            name = normalize_template_name(part.name)
            index = used.get(name, 0)
            used[name] = index + 1
            candidates = original_by_name.get(name, [])
            if index < len(candidates):
                result.append(candidates[index])
                continue
        result.append(str(part))
    return "".join(result)


def save_page(page: Page | str, text: str, summary: str = "update page page"):
    if isinstance(page, str):
        page = Page(s, page)
    ignore_page, ignored_templates = get_ignore_rules(page.title())
    if ignore_page:
        print(f"Ignored {page.title()} (in ignore_list.json)")
        return
    original = page.text
    text = keep_ignored_templates(original, text, ignored_templates)
    if original.strip() == text.strip():
        return
    print(f"Changes to {page.title()}:")
    showDiff(original, text)
    page.text = text
    page.save(summary=summary)


if __name__ == "__main__":
    raise NotImplementedError("Do not run this script directly.")