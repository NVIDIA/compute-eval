import gzip
import importlib.resources as pkg_resources
import os
from collections.abc import Collection
from pathlib import Path

import yaml
from tree_sitter import Language, Node, Parser
from tree_sitter_language_pack import get_language, get_parser

# (tree sitter name, linguist name, codemirror mime type, popularity rank) tuples.
# Source of tree sitter names:
# https://github.com/Goldziher/tree-sitter-language-pack
# Source of linguist names:
# https://raw.githubusercontent.com/github-linguist/linguist/master/lib/linguist/languages.yml
# (mirrored in encoding/language_data/languages.yml)
# Source of CodeMirror MIME types:
# From linguist languages.yml codemirror_mime_type field
# Source of popularity ranks:
# https://innovationgraph.github.com/global-metrics/programming-languages
# https://raw.githubusercontent.com/github/innovationgraph/main/data/languages.csv
RAW_LANGUAGE_DATA = [
    ("actionscript", "ActionScript", None, 159),
    ("ada", "Ada", None, 138),
    ("agda", "Agda", None, 270),
    ("apex", "Apex", "text/x-java", 154),
    ("asm", "Assembly", None, 27),
    ("astro", "Astro", "text/jsx", 89),
    ("bash", "Shell", "text/x-sh", 5),
    ("bibtex", "BibTeX", "text/x-stex", 300),
    ("bicep", "Bicep", None, 135),
    ("bitbake", "BitBake", None, 139),
    ("c", "C", "text/x-csrc", 11),
    ("cairo", "Cairo", None, 262),
    ("capnp", "Cap'n Proto", None, 194),
    ("clarity", "Clarity", None, 365),
    ("clojure", "Clojure", "text/x-clojure", 86),
    ("cmake", "CMake", "text/x-cmake", 18),
    ("commonlisp", "Common Lisp", "text/x-common-lisp", 97),
    ("cpp", "C++", "text/x-c++src", 9),
    ("csharp", "C#", "text/x-csharp", 15),
    ("css", "CSS", "text/css", 3),
    ("csv", "CSV", None, 400),
    ("cuda", "Cuda", "text/x-c++src", 62),
    ("d", "D", "text/x-d", 115),
    ("dart", "Dart", "application/dart", 29),
    ("dockerfile", "Dockerfile", "text/x-dockerfile", 8),
    ("elisp", "Emacs Lisp", "text/x-common-lisp", 56),
    ("elixir", "Elixir", None, 94),
    ("elm", "Elm", "text/x-elm", 158),
    ("erlang", "Erlang", "text/x-erlang", 109),
    ("fennel", "Fennel", None, 333),
    ("firrtl", "FIRRTL", None, 750),
    ("fish", "fish", None, 220),
    ("fortran", "Fortran Free Form", "text/x-fortran", 355),
    ("fsharp", "F#", "text/x-fsharp", 116),
    ("gdscript", "GDScript", None, 98),
    ("gitattributes", "Git Attributes", "text/x-sh", 600),
    ("gitcommit", "Git Commit", None, 1000),
    ("gitignore", "Ignore List", "text/x-sh", 500),
    ("gleam", "Gleam", None, 328),
    ("glsl", "GLSL", None, 46),
    ("gn", "GN", "text/x-python", 500),
    ("go", "Go", "text/x-go", 20),
    ("gomod", "Go Module", None, 355),
    ("gosum", "Go Checksums", None, 600),
    ("graphql", "GraphQL", None, 130),
    ("groovy", "Groovy", "text/x-groovy", 52),
    ("hack", "Hack", "application/x-httpd-php", 31),
    ("hare", "Hare", None, 800),
    ("haskell", "Haskell", "text/x-haskell", 73),
    ("haxe", "Haxe", "text/x-haxe", 166),
    ("hcl", "HCL", "text/x-ruby", 38),
    ("heex", "HTML+EEX", "text/html", 400),
    ("hlsl", "HLSL", None, 41),
    ("html", "HTML", "text/html", 1),
    ("ispc", "ISPC", "text/x-csrc", 550),
    ("ini", "INI", "text/x-properties", 300),
    ("janet", "Janet", "text/x-scheme", 375),
    ("java", "Java", "text/x-java", 6),
    ("javascript", "JavaScript", "text/javascript", 2),
    ("json", "JSON", "application/json", 190),
    ("jsonnet", "Jsonnet", None, 127),
    ("julia", "Julia", "text/x-julia", 85),
    ("kdl", "KDL", "text/x-yacas", 800),
    ("kotlin", "Kotlin", "text/x-kotlin", 19),
    ("latex", "TeX", "text/x-stex", 30),
    ("linkerscript", "Linker Script", None, 201),
    ("llvm", "LLVM", None, 126),
    ("lua", "Lua", "text/x-lua", 25),
    ("luau", "Luau", "text/x-lua", 300),
    ("make", "Makefile", "text/x-cmake", 12),
    ("markdown", "Markdown", "text/x-gfm", 146),
    ("matlab", "MATLAB", "text/x-octave", 45),
    ("mermaid", "Mermaid", None, 192),
    ("meson", "Meson", None, 79),
    ("netlinx", "NetLinx", None, 1000),
    ("nim", "Nim", None, 140),
    ("ninja", "Ninja", None, 450),
    ("nix", "Nix", None, 50),
    ("objc", "Objective-C", "text/x-objectivec", 21),
    ("ocaml", "OCaml", "text/x-ocaml", 107),
    ("odin", "Odin", None, 314),
    ("org", "Org", None, 320),
    ("pascal", "Pascal", "text/x-pascal", 77),
    ("perl", "Perl", "text/x-perl", 28),
    ("php", "PHP", "application/x-httpd-php", 14),
    ("po", "Gettext Catalog", None, 500),
    ("pony", "Pony", None, 326),
    ("powershell", "PowerShell", "application/x-powershell", 24),
    ("prisma", "Prisma", None, 220),
    ("properties", "INI", "text/x-properties", 300),
    ("proto", "Protocol Buffer", "text/x-protobuf", 340),
    ("puppet", "Puppet", "text/x-puppet", 133),
    ("purescript", "PureScript", "text/x-haskell", 165),
    ("python", "Python", "text/x-python", 4),
    ("qmljs", "QML", None, 114),
    ("query", "Tree-sitter Query", None, 1000),
    ("r", "R", "text/x-rsrc", 32),
    ("racket", "Racket", None, 170),
    ("re2c", "RenderScript", None, 141),
    ("readline", "Readline Config", None, 1000),
    ("rego", "Open Policy Agent", None, 172),
    ("requirements", "Pip Requirements", None, 400),
    ("ron", "RON", None, 650),
    ("rst", "reStructuredText", "text/x-rst", 374),
    ("ruby", "Ruby", "text/x-ruby", 16),
    ("rust", "Rust", "text/x-rustsrc", 26),
    ("scala", "Scala", "text/x-scala", 59),
    ("scheme", "Scheme", "text/x-scheme", 84),
    ("scss", "SCSS", "text/x-scss", 10),
    ("smali", "Smali", None, 351),
    ("smithy", "Smithy", "text/x-csrc", 353),
    ("solidity", "Solidity", None, 57),
    ("sparql", "SPARQL", "application/sparql-query", 600),
    ("sql", "SQL", "text/x-sql", 339),
    ("squirrel", "Squirrel", "text/x-squirrel", 299),
    ("starlark", "Starlark", "text/x-python", 51),
    ("svelte", "Svelte", "text/html", 67),
    ("swift", "Swift", "text/x-swift", 23),
    ("tcl", "Tcl", "text/x-tcl", 68),
    ("thrift", "Thrift", None, 130),
    ("toml", "TOML", "text/x-toml", 355),
    ("tsv", "TSV", None, 550),
    ("tsx", "TSX", "text/typescript-jsx", 6),
    ("twig", "Twig", "text/x-twig", 72),
    ("typescript", "TypeScript", "application/typescript", 7),
    ("typst", "Typst", None, 500),
    ("v", "V", "text/x-go", 205),
    ("verilog", "Verilog", "text/x-verilog", 92),
    ("vhdl", "VHDL", "text/x-vhdl", 113),
    ("vim", "Vim Script", None, 37),
    ("vue", "Vue", "text/x-vue", 22),
    ("wast", "WebAssembly", "text/webassembly", 160),
    ("wat", "WebAssembly", "text/webassembly", 160),
    ("wgsl", "WGSL", None, 550),
    ("xcompose", "XCompose", None, 900),
    ("xml", "XML", "text/xml", 92),
    ("yaml", "YAML", "text/x-yaml", 180),
    ("zig", "Zig", None, 149),
]

RESOURCE_PATH = "language_data"
LINGUIST_FILE_NAME = "languages.yml.gz"


class ParseableLanguage:
    """
    Represents a programming language that can be parsed using tree-sitter.

    This class encapsulates language metadata (names, file extensions, known files)
    and provides parsing capabilities through tree-sitter. It lazily initializes
    the tree-sitter language and parser objects on first use.
    """

    def __init__(
        self,
        canonical_name: str,
        treesitter_name: str,
        file_extensions: Collection[str],
        alt_file_extensions: Collection[str],
        known_files: Collection[str],
        alt_known_files: Collection[str],
        popularity_rank: int,
    ):
        """
        Initialize a ParseableLanguage instance.

        Args:
            canonical_name: The canonical/linguist name for the language (e.g., "Python", "C++").
            treesitter_name: The tree-sitter language identifier (e.g., "python", "cpp").
            file_extensions: Primary file extensions for this language (e.g., [".py", ".pyw"]).
            alt_file_extensions: Alternative/secondary file extensions.
            known_files: Known filenames that identify this language (e.g., ["Makefile"]).
            alt_known_files: Alternative known filenames.
            popularity_rank: Numeric rank indicating language popularity (lower is more popular).
        """
        self.canonical_name = canonical_name
        self.treesitter_name = treesitter_name
        self.file_extensions = frozenset(file_extensions)
        self.alt_file_extensions = frozenset(alt_file_extensions)
        self.known_files = frozenset(known_files)
        self.alt_known_files = frozenset(alt_known_files)
        self.popularity_rank = popularity_rank
        self._language = None
        self._parser = None

    def get_language(self) -> Language:
        """
        Get the tree-sitter Language object for this language.

        The Language object is lazily initialized on first access and cached
        for subsequent calls.

        Returns:
            The tree-sitter Language object for parsing this language.
        """
        if self._language is None:
            self._language = get_language(self.treesitter_name)
        return self._language

    def get_parser(self) -> Parser:
        """
        Get the tree-sitter Parser object for this language.

        The Parser object is lazily initialized on first access and cached
        for subsequent calls.

        Returns:
            The tree-sitter Parser object configured for this language.
        """
        if self._parser is None:
            self._parser = get_parser(self.treesitter_name)
        return self._parser

    @staticmethod
    def _accumulate_errors(node: Node, errors: list[tuple[int, int]]):
        if node.is_error:
            errors.append((node.start_point[0] + 1, node.end_point[0] + 1))
        elif node.has_error:
            for child in node.children:
                ParseableLanguage._accumulate_errors(child, errors)

    def parse_errors(self, contents: bytes) -> list[tuple[int, int]]:
        """
        Parse the given contents and return a list of syntax errors.

        Each error is represented as a tuple of (start_line, end_line) where
        line numbers are 1-indexed.

        Args:
            contents: The source code to parse as bytes.

        Returns:
            A list of tuples (start_line, end_line) representing error ranges.
            Returns an empty list if there are no syntax errors.
        """
        tree = self.get_parser().parse(contents)
        errors = []
        self._accumulate_errors(tree.root_node, errors)
        return errors

    def parse_error_line_count(self, contents: bytes) -> int:
        """
        Count the total number of lines containing syntax errors.

        This method parses the contents and calculates the total number of lines
        that are part of error ranges (inclusive of start and end lines).

        Args:
            contents: The source code to parse as bytes.

        Returns:
            The total count of lines with syntax errors.
        """
        error_lines = 0
        for start_line, end_line in self.parse_errors(contents):
            error_lines += end_line - start_line + 1
        return error_lines

    def parse_tree_spans(self, contents: bytes) -> dict[tuple[int, int], str]:
        """
        Parse contents and return a mapping of byte spans to node types.

        This method creates a dictionary mapping byte ranges to their corresponding
        tree-sitter node types. Only includes complete, non-error nodes that are not
        "extra" (comments, whitespace, etc.) or missing. Uses tree.walk() for
        efficient linear-time traversal via native C code.

        Args:
            contents: The source code to parse as bytes.

        Returns:
            A dictionary mapping (start_byte, end_byte) tuples to node type strings.
            The byte positions are 0-indexed offsets into the contents.
        """
        tree = self.get_parser().parse(contents)
        spans = {}

        cursor = tree.walk()
        visited_children = False

        # preorder: matches a left-to-right, top-to-bottom scan of the file
        while True:
            node = cursor.node
            if not node.is_error and not visited_children:
                if not node.is_extra and not node.is_missing:
                    spans[(node.start_byte, node.end_byte)] = node.type
                if cursor.goto_first_child():
                    visited_children = False
                    continue

            if cursor.goto_next_sibling():
                visited_children = False
                continue
            if not cursor.goto_parent():
                break
            visited_children = True

        return spans

    @staticmethod
    def _find_all(content: bytes, target: bytes) -> list[int]:
        matches = []
        start_byte = 0
        while True:
            start_byte = content.find(target, start_byte)
            if start_byte == -1:
                break
            matches.append(start_byte)
            start_byte += len(target)
        return matches

    def find_matching_subtrees(self, content: bytes, targets: list[bytes]) -> list[tuple[bytes, int]]:
        """
        Find exact byte sequence matches that correspond to complete parse tree nodes.

        This method searches for exact byte sequences in the content and verifies that
        each match corresponds to a complete, valid subtree in the parse tree (not an
        error, extra, or missing node, and with exact byte boundaries).

        Args:
            content: The source code to search in as bytes.
            targets: A list of byte sequences to search for.

        Returns:
            A list of tuples (target, byte_offset) where:
            - target: The matched byte sequence from the targets list
            - byte_offset: The 0-indexed byte position where the match starts
            Only returns matches that are valid complete subtrees.
        """
        tree = self.get_parser().parse(content)
        root = tree.root_node
        matches = []
        for target in targets:
            target_len = len(target)
            for match in self._find_all(content, target):
                subtree = root.descendant_for_byte_range(match, match + target_len)
                if (
                    subtree is not None
                    and not subtree.is_error
                    and not subtree.is_extra
                    and not subtree.is_missing
                    and subtree.start_byte == match
                    and subtree.end_byte == match + target_len
                ):
                    matches.append((target, match))

        return matches

    def __str__(self):
        return self.canonical_name

    def __repr__(self):
        return f"ParseableLanguage({self.canonical_name})"

    def __hash__(self):
        return hash(self.canonical_name)

    def __eq__(self, other):
        if not isinstance(other, ParseableLanguage):
            return False
        return self.canonical_name == other.canonical_name

    def __ne__(self, other):
        return not self.__eq__(other)


def _collect_exts_and_files(linguist_data, ling_language, cm_mime):
    file_extensions, alt_file_extensions = set(), set()
    known_files, alt_known_files = set(), set()
    for language, data in linguist_data.items():
        if language == ling_language:
            file_extensions.update(data.get("extensions", []))
            known_files.update(data.get("filenames", []))
        else:
            if ling_language and data.get("group") == ling_language:
                alt_file_extensions.update(data.get("extensions", []))
                alt_known_files.update(data.get("filenames", []))
            if cm_mime and data.get("codemirror_mime_type") == cm_mime:
                alt_file_extensions.update(data.get("extensions", []))
                alt_known_files.update(data.get("filenames", []))
    return file_extensions, alt_file_extensions, known_files, alt_known_files


def _build_parseable_languages() -> list[ParseableLanguage]:
    try:
        if __package__:
            resources = pkg_resources.files(__package__) / RESOURCE_PATH
        else:
            resources = Path(__file__).parent / RESOURCE_PATH
    except Exception:
        resources = Path(__file__).parent / RESOURCE_PATH

    linguist_file = resources / LINGUIST_FILE_NAME
    with gzip.open(linguist_file, "rt") as f:
        linguist_data = yaml.safe_load(f)

    parseable_languages = []

    for ts_language, ling_language, cm_mime, popularity_rank in RAW_LANGUAGE_DATA:
        if ling_language is not None:
            file_extensions, alt_file_extensions, known_files, alt_known_files = _collect_exts_and_files(
                linguist_data, ling_language, cm_mime
            )

            parseable_languages.append(
                ParseableLanguage(
                    canonical_name=ling_language,
                    treesitter_name=ts_language,
                    file_extensions=file_extensions,
                    alt_file_extensions=alt_file_extensions,
                    known_files=known_files,
                    alt_known_files=alt_known_files,
                    popularity_rank=popularity_rank,
                )
            )

    return parseable_languages


def _build_file_and_extension_maps(
    languages: list[ParseableLanguage],
) -> tuple[
    dict[str, set[ParseableLanguage]],
    dict[str, set[ParseableLanguage]],
    dict[str, set[ParseableLanguage]],
    dict[str, set[ParseableLanguage]],
]:
    extension_map, alt_extension_map = {}, {}
    file_map, alt_file_map = {}, {}
    for language in languages:
        for extension in language.file_extensions:
            extension_map.setdefault(extension, set()).add(language)
        for extension in language.alt_file_extensions:
            alt_extension_map.setdefault(extension, set()).add(language)
        for filename in language.known_files:
            file_map.setdefault(filename, set()).add(language)
        for filename in language.alt_known_files:
            alt_file_map.setdefault(filename, set()).add(language)
    return extension_map, alt_extension_map, file_map, alt_file_map


ALL_LANGUAGES = _build_parseable_languages()

(
    LANGUAGES_BY_EXTENSION,
    LANGUAGES_BY_ALT_EXTENSION,
    LANGUAGES_BY_FILENAME,
    LANGUAGES_BY_ALT_FILENAME,
) = _build_file_and_extension_maps(ALL_LANGUAGES)


def _get_languages_by_path(
    path: str,
    ext_map: dict[str, set[ParseableLanguage]],
    fname_map: dict[str, set[ParseableLanguage]],
) -> Collection[ParseableLanguage]:
    parts = path.split(os.sep)
    if not parts:
        return []

    file = parts[-1]
    if file in fname_map:
        return fname_map[file]

    ext_idx = file.find(".")
    while ext_idx != -1:
        extension = file[ext_idx:]
        if extension in ext_map:
            return ext_map[extension]
        ext_idx = file.find(".", ext_idx + 1)
    return []


def get_language_by_path(path: str) -> ParseableLanguage | None:
    languages = _get_languages_by_path(path, LANGUAGES_BY_EXTENSION, LANGUAGES_BY_FILENAME)
    if not languages:
        languages = _get_languages_by_path(path, LANGUAGES_BY_ALT_EXTENSION, LANGUAGES_BY_ALT_FILENAME)
    return min(languages, key=lambda lang: lang.popularity_rank, default=None)


def _get_most_likely_language(
    path: str,
    contents: bytes,
    ext_map: dict[str, set[ParseableLanguage]],
    fname_map: dict[str, set[ParseableLanguage]],
) -> ParseableLanguage | None:
    languages = _get_languages_by_path(path, ext_map, fname_map)
    if len(languages) == 1:
        return next(iter(languages))
    elif len(languages) > 1:
        return min(
            languages,
            key=lambda lang: (
                lang.parse_error_line_count(contents),
                lang.popularity_rank,
            ),
        )
    return None


def get_most_likely_language(path: str, contents: bytes | str) -> ParseableLanguage | None:
    if isinstance(contents, str):
        contents = contents.encode("utf-8")
    language = _get_most_likely_language(path, contents, LANGUAGES_BY_EXTENSION, LANGUAGES_BY_FILENAME)
    if language is None:
        language = _get_most_likely_language(path, contents, LANGUAGES_BY_ALT_EXTENSION, LANGUAGES_BY_ALT_FILENAME)
    return language


if __name__ == "__main__":
    for language in ALL_LANGUAGES:
        if language.get_language() is None or language.get_parser() is None:
            print(f"Failed to load tree-sitter language or parser for {language}")
            exit(1)

    sample_file = Path(__file__).parent.parent / "temp/cudaTestProgram.cu"
    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        print("Please provide a valid file path to test the parser.")
        exit(1)

    with open(sample_file, "rb") as f:
        sample_contents = f.read()

    language = get_most_likely_language(str(sample_file), sample_contents)
    if language is None:
        print(f"Could not determine language for {sample_file}")
        exit(1)

    print(f"Detected language: {language}")
    print(f"Parsing {sample_file}...")
    errors = language.parse_errors(sample_contents)
    if errors:
        print(f"Parse errors: {errors}")
    else:
        print("No parse errors")

    matches = language.find_matching_subtrees(sample_contents, [b"cudaCheckErrors", b"cuda", b"d_", b"d_A"])
    if matches:
        print(f"Found matches: {matches}")
    else:
        print("No matches found")
