"""Tests for the multi-format ingestion layer.

The claim this file exists to check is narrow and load-bearing: **the format a
dataset arrived in stops mattering at the adapter**. Everything above the
adapter is about recognising and refusing; everything below it works on one
standardised table.

So the tests come in three groups. The adapters are exercised on their own —
each format's happy path, and each format's ways of being broken. The registry
and detection are exercised as the allowlist they are. And the three are
compared: the same logical data, written three ways, must produce the same
frame and the same fingerprint, because that is what "format-agnostic" has to
mean if it is to mean anything.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.core.errors import (
    DatasetTooLargeError,
    DuplicateColumnsError,
    EmptyDatasetError,
    InvalidDatasetContentError,
    InvalidExcelError,
    InvalidJSONError,
    MalformedCSVError,
    MissingHeaderError,
    UnsupportedFileTypeError,
)
from app.services.datasets import DatasetProfilingService
from app.services.datasets.ingestion import (
    CSVAdapter,
    DatasetAdapter,
    DatasetAdapterRegistry,
    DatasetFormat,
    ExcelAdapter,
    JSONAdapter,
    default_registry,
    detect_format,
    supported_formats,
)
from app.services.datasets.ingestion.normalisation import looks_binary
from ml.experiments.fingerprint import fingerprint_dataset
from tests.factories import (
    FakeUpload,
    build_csv,
    build_json,
    build_xlsx,
    csv_as_json,
    csv_as_xlsx,
    frame_from_csv,
    frame_to_json,
    frame_to_xlsx,
    multi_sheet_xlsx,
    sample_csv,
)

HEADER = ["age", "income", "city", "score"]
ROWS: list[list[object]] = [
    [20, 50_000, "Paris", 1.5],
    [30, 70_000, "Lyon", 2.5],
    [40, 90_000, "Nice", 3.5],
]


class Upload(FakeUpload):
    """A fake upload that also declares a media type, as a browser does."""

    def __init__(self, content: bytes, content_type: str | None = None) -> None:
        """Store the bytes and the media type the client would have sent."""
        super().__init__(content)
        self.content_type = content_type


# ---------------------------------------------------------------------------
# 1. The CSV adapter
# ---------------------------------------------------------------------------


def test_the_csv_adapter_reads_a_well_formed_file(settings: Settings) -> None:
    """CSV bytes become a frame with the file's own columns and shape."""
    frame = CSVAdapter().load(build_csv(HEADER, ROWS), settings)

    assert list(frame.columns) == HEADER
    assert frame.shape == (3, 4)


def test_the_csv_adapter_reports_its_identity() -> None:
    """An adapter names the one format it reads, and only that one."""
    adapter = CSVAdapter()

    assert adapter.format is DatasetFormat.CSV
    assert adapter.format_name == "csv"
    assert adapter.can_handle(DatasetFormat.CSV) is True
    assert adapter.can_handle(DatasetFormat.JSON) is False


def test_every_adapter_satisfies_the_protocol() -> None:
    """The three adapters are interchangeable as far as callers are concerned."""
    for adapter in (CSVAdapter(), ExcelAdapter(), JSONAdapter()):
        assert isinstance(adapter, DatasetAdapter)


# ---------------------------------------------------------------------------
# 2. The Excel adapter
# ---------------------------------------------------------------------------


def test_the_excel_adapter_reads_a_workbook(settings: Settings) -> None:
    """A one-sheet workbook becomes the same frame the CSV would have."""
    frame = ExcelAdapter().load(build_xlsx(HEADER, ROWS), settings)

    assert list(frame.columns) == HEADER
    assert frame.shape == (3, 4)


def test_the_excel_adapter_reads_the_first_worksheet(settings: Settings) -> None:
    """With several sheets the first is read, and the others are ignored.

    Documented behaviour, not an accident: a workbook is not a table, so
    something has to choose, and the choice is stated in every README.
    """
    workbook = multi_sheet_xlsx(
        {
            "first": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "second": pd.DataFrame({"totally": ["different"], "shape": [1]}),
        }
    )

    frame = ExcelAdapter().load(workbook, settings)

    assert list(frame.columns) == ["a", "b"]
    assert frame.shape == (2, 2)


def test_the_excel_adapter_preserves_missing_values(settings: Settings) -> None:
    """An empty cell arrives as a missing value, not as an empty string."""
    frame = ExcelAdapter().load(
        build_xlsx(["a", "b"], [[1, None], [2, "x"]]), settings
    )

    assert frame["b"].isna().tolist() == [True, False]


def test_the_excel_adapter_reports_its_identity() -> None:
    """The Excel adapter reads ``.xlsx`` and nothing else."""
    adapter = ExcelAdapter()

    assert adapter.format is DatasetFormat.XLSX
    assert adapter.format_name == "xlsx"
    assert adapter.can_handle(DatasetFormat.CSV) is False


# ---------------------------------------------------------------------------
# 3. The JSON adapter
# ---------------------------------------------------------------------------


def test_the_json_adapter_reads_an_array_of_objects(settings: Settings) -> None:
    """The canonical shape — one object per row — becomes a table."""
    frame = JSONAdapter().load(build_json(HEADER, ROWS), settings)

    assert list(frame.columns) == HEADER
    assert frame.shape == (3, 4)


def test_the_json_adapter_reads_records_under_an_envelope(
    settings: Settings,
) -> None:
    """``{"rows": [...]}`` is the shape most APIs actually return."""
    document = json.dumps(
        {"rows": [{"a": 1, "b": 2}, {"a": 3, "b": 4}], "count": 2}
    ).encode("utf-8")

    frame = JSONAdapter().load(document, settings)

    assert list(frame.columns) == ["a", "b"]
    assert frame.shape == (2, 2)


def test_the_json_adapter_breaks_a_tie_with_a_conventional_key(
    settings: Settings,
) -> None:
    """Two candidate arrays are ambiguous unless one is a conventional name."""
    document = json.dumps(
        {"data": [{"a": 1}, {"a": 2}], "errors": [{"message": "none"}]}
    ).encode("utf-8")

    frame = JSONAdapter().load(document, settings)

    assert list(frame.columns) == ["a"]


def test_the_json_adapter_refuses_two_unconventional_arrays(
    settings: Settings,
) -> None:
    """When it cannot tell which array is the data, it says so."""
    document = json.dumps(
        {"alpha": [{"a": 1}], "beta": [{"b": 2}]}
    ).encode("utf-8")

    with pytest.raises(InvalidJSONError) as exc_info:
        JSONAdapter().load(document, settings)
    assert "more than one list of records" in str(exc_info.value)


def test_the_json_adapter_reads_a_single_flat_object(settings: Settings) -> None:
    """One object of scalars is one row, which is a table with one row in it."""
    frame = JSONAdapter().load(b'{"a": 1, "b": "x"}', settings)

    assert frame.shape == (1, 2)


def test_the_json_adapter_flattens_a_nested_object(settings: Settings) -> None:
    """A nested object becomes dotted column names — the only tabular reading."""
    document = json.dumps(
        [{"id": 1, "address": {"city": "Paris"}}, {"id": 2, "address": {"city": "Lyon"}}]
    ).encode("utf-8")

    frame = JSONAdapter().load(document, settings)

    assert list(frame.columns) == ["id", "address.city"]


def test_the_json_adapter_preserves_nulls_as_missing_values(
    settings: Settings,
) -> None:
    """A JSON ``null`` is a missing value, exactly as a blank CSV field is."""
    frame = JSONAdapter().load(b'[{"a": 1, "b": null}, {"a": 2, "b": 5}]', settings)

    assert frame["b"].isna().tolist() == [True, False]


def test_the_json_adapter_reports_its_identity() -> None:
    """The JSON adapter reads JSON and nothing else."""
    adapter = JSONAdapter()

    assert adapter.format is DatasetFormat.JSON
    assert adapter.format_name == "json"


# ---------------------------------------------------------------------------
# 4. The adapter registry
# ---------------------------------------------------------------------------


def test_the_default_registry_holds_exactly_three_formats() -> None:
    """The readable formats are the ones registered, and they are these three.

    Parquet, SQL, databases and remote sources are deliberately absent.
    """
    registry = default_registry()

    assert registry.format_names() == ("csv", "xlsx", "json")
    assert registry.extensions() == (".csv", ".xlsx", ".json")


def test_the_registry_returns_the_adapter_for_a_format() -> None:
    """Lookup is by format, never by filename or by string name."""
    registry = default_registry()

    assert isinstance(registry.adapter_for(DatasetFormat.XLSX), ExcelAdapter)
    assert isinstance(registry.adapter_for(DatasetFormat.JSON), JSONAdapter)


def test_an_unregistered_format_is_refused_as_unsupported() -> None:
    """A registry without an adapter refuses rather than failing obscurely."""
    registry = DatasetAdapterRegistry([CSVAdapter()])

    assert registry.supports(DatasetFormat.XLSX) is False
    with pytest.raises(UnsupportedFileTypeError):
        registry.adapter_for(DatasetFormat.XLSX)


def test_the_registry_reports_metadata_without_any_rows(
    settings: Settings,
) -> None:
    """Metadata describes the dataset and carries none of it."""
    ingested = default_registry().load(
        build_json(HEADER, ROWS), DatasetFormat.JSON, settings, filename="people.json"
    )
    payload = ingested.metadata.as_dict()

    assert payload == {
        "source_format": "json",
        "row_count": 3,
        "column_count": 4,
        "original_filename": "people.json",
    }
    for row in ROWS:
        for value in row:
            assert str(value) not in json.dumps(payload)


def test_registering_an_adapter_replaces_the_previous_one() -> None:
    """A registry is an allowlist, so one format has exactly one reader."""
    registry = DatasetAdapterRegistry([CSVAdapter(), CSVAdapter()])

    assert registry.format_names() == ("csv",)


# ---------------------------------------------------------------------------
# 5. Format detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("data.csv", DatasetFormat.CSV),
        ("DATA.CSV", DatasetFormat.CSV),
        ("report.xlsx", DatasetFormat.XLSX),
        ("records.json", DatasetFormat.JSON),
    ],
)
def test_detection_reads_the_extension(
    filename: str, expected: DatasetFormat, settings: Settings
) -> None:
    """The extension is the primary signal, and case does not matter."""
    assert detect_format(filename, settings) is expected


def test_detection_falls_back_to_the_media_type(settings: Settings) -> None:
    """With no extension at all, the declared media type is the only signal."""
    assert detect_format("dataset", settings, "application/json") is DatasetFormat.JSON
    assert (
        detect_format("dataset", settings, "text/csv; charset=utf-8")
        is DatasetFormat.CSV
    )


def test_the_media_type_never_overrides_the_extension(settings: Settings) -> None:
    """A mislabelled upload is still read as what its extension claims.

    The media type comes from the same untrusted client as the filename, so it
    is a stand-in for a missing extension and never a correction to one. What
    protects the system is not detection but the adapter, which validates the
    bytes.
    """
    assert (
        detect_format("data.csv", settings, "application/json") is DatasetFormat.CSV
    )


def test_detection_permits_only_the_configured_formats() -> None:
    """The allowlist is the existing settings field, not a new one."""
    narrowed = Settings(supported_dataset_extensions=(".csv",))

    assert supported_formats(narrowed) == (DatasetFormat.CSV,)
    with pytest.raises(UnsupportedFileTypeError):
        detect_format("data.xlsx", narrowed)


# ---------------------------------------------------------------------------
# 6. Unsupported formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["data.parquet", "data.txt", "data.xls", "data.xlsm", "data", "data.csv.exe"],
)
def test_an_unsupported_extension_is_refused(
    filename: str, settings: Settings
) -> None:
    """Parquet, SQL dumps, macros and bare names are all simply not readable."""
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        detect_format(filename, settings)

    error = exc_info.value
    assert error.status_code == 415
    assert error.details["supported_formats"] == ["csv", "xlsx", "json"]


# ---------------------------------------------------------------------------
# 7-9. Malformed content, per format
# ---------------------------------------------------------------------------


def test_malformed_csv_is_still_refused(settings: Settings) -> None:
    """The CSV failure modes from Commit 2 survive the refactor unchanged."""
    with pytest.raises(MalformedCSVError):
        CSVAdapter().load(b"a,b\n1,2\n3,4,5\n", settings)


def test_a_csv_that_repeats_a_column_is_refused(settings: Settings) -> None:
    """Duplicate headers are refused rather than silently renamed."""
    with pytest.raises(DuplicateColumnsError):
        CSVAdapter().load(b"a,a\n1,2\n", settings)


def test_a_file_named_xlsx_that_is_not_a_workbook_fails(
    settings: Settings,
) -> None:
    """The extension chose the reader; the bytes decide the outcome."""
    with pytest.raises(InvalidExcelError) as exc_info:
        ExcelAdapter().load(b"age,income\n20,50000\n", settings)

    assert exc_info.value.code == "invalid_excel"
    assert exc_info.value.status_code == 422


def test_a_truncated_workbook_fails_cleanly(settings: Settings) -> None:
    """Corrupt workbook bytes produce a message, never a reader traceback."""
    workbook = build_xlsx(HEADER, ROWS)

    with pytest.raises(InvalidExcelError) as exc_info:
        ExcelAdapter().load(workbook[:200], settings)

    message = str(exc_info.value)
    assert "Traceback" not in message
    assert "openpyxl" not in message.lower()
    assert "zipfile" not in message.lower()


def test_malformed_json_is_refused(settings: Settings) -> None:
    """Broken syntax is reported with the position, and nothing else."""
    with pytest.raises(InvalidJSONError) as exc_info:
        JSONAdapter().load(b'[{"a": 1},', settings)

    assert exc_info.value.code == "invalid_json"
    assert "line" in str(exc_info.value)


def test_a_file_named_json_holding_csv_fails(settings: Settings) -> None:
    """A mislabelled CSV is refused by the JSON adapter, not half-parsed."""
    with pytest.raises(InvalidJSONError):
        JSONAdapter().load(b"age,income\n20,50000\n", settings)


@pytest.mark.parametrize("document", [b"42", b'"text"', b"true", b"null"])
def test_scalar_json_is_not_a_dataset(document: bytes, settings: Settings) -> None:
    """A single JSON value is refused, however valid the JSON is."""
    with pytest.raises(InvalidJSONError):
        JSONAdapter().load(document, settings)


def test_a_json_array_of_scalars_is_refused(settings: Settings) -> None:
    """An array must hold objects; a list of numbers has no columns."""
    with pytest.raises(InvalidJSONError):
        JSONAdapter().load(b"[1, 2, 3]", settings)


def test_json_records_holding_arrays_are_refused(settings: Settings) -> None:
    """A cell cannot hold a list, and quietly stringifying one would lie."""
    document = json.dumps([{"a": 1, "tags": ["x", "y"]}, {"a": 2, "tags": []}])

    with pytest.raises(InvalidJSONError) as exc_info:
        JSONAdapter().load(document.encode("utf-8"), settings)

    assert exc_info.value.details["nested_fields"] == ["tags"]


def test_deeply_nested_json_is_refused(settings: Settings) -> None:
    """An arbitrary tree is not a table and is not guessed at."""
    document = json.dumps({"a": {"b": {"c": [1, 2, 3]}}})

    with pytest.raises(InvalidJSONError):
        JSONAdapter().load(document.encode("utf-8"), settings)


# ---------------------------------------------------------------------------
# 10-12. Empty content, per format
# ---------------------------------------------------------------------------


def test_an_empty_csv_is_refused(settings: Settings) -> None:
    """No bytes at all means no dataset."""
    with pytest.raises((EmptyDatasetError, MissingHeaderError)):
        CSVAdapter().load(b"", settings)


def test_a_header_only_csv_is_refused(settings: Settings) -> None:
    """Columns with no rows is the other kind of empty."""
    with pytest.raises(EmptyDatasetError):
        CSVAdapter().load(b"a,b\n", settings)


def test_an_empty_worksheet_is_refused(settings: Settings) -> None:
    """A workbook whose first sheet holds nothing has nothing to profile."""
    with pytest.raises(EmptyDatasetError):
        ExcelAdapter().load(frame_to_xlsx(pd.DataFrame()), settings)


def test_a_header_only_worksheet_is_refused(settings: Settings) -> None:
    """A sheet with column titles and no data is refused like an empty CSV."""
    with pytest.raises(EmptyDatasetError):
        ExcelAdapter().load(
            frame_to_xlsx(pd.DataFrame(columns=["a", "b"])), settings
        )


@pytest.mark.parametrize("document", [b"[]", b"{}", b"   "])
def test_empty_json_is_refused(document: bytes, settings: Settings) -> None:
    """An empty array, an empty object and whitespace all hold no rows."""
    with pytest.raises(EmptyDatasetError):
        JSONAdapter().load(document, settings)


# ---------------------------------------------------------------------------
# 13. Shared limits, applied identically
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_format", ["csv", "xlsx", "json"])
def test_every_format_obeys_the_same_row_limit(source_format: str) -> None:
    """One row limit, applied by all three adapters. No per-format limits."""
    narrow = Settings(max_dataset_rows=2)
    rows: list[list[object]] = [[index, index * 2] for index in range(5)]
    payloads = {
        "csv": (CSVAdapter(), build_csv(["a", "b"], rows)),
        "xlsx": (ExcelAdapter(), build_xlsx(["a", "b"], rows)),
        "json": (JSONAdapter(), build_json(["a", "b"], rows)),
    }
    adapter, content = payloads[source_format]

    with pytest.raises(DatasetTooLargeError) as exc_info:
        adapter.load(content, narrow)
    assert exc_info.value.status_code == 413


@pytest.mark.parametrize("source_format", ["csv", "xlsx", "json"])
def test_every_format_obeys_the_same_column_limit(source_format: str) -> None:
    """The column limit is shared too, and is checked before parsing rows."""
    narrow = Settings(max_dataset_columns=2)
    header = ["a", "b", "c"]
    rows: list[list[object]] = [[1, 2, 3]]
    payloads = {
        "csv": (CSVAdapter(), build_csv(header, rows)),
        "xlsx": (ExcelAdapter(), build_xlsx(header, rows)),
        "json": (JSONAdapter(), build_json(header, rows)),
    }
    adapter, content = payloads[source_format]

    with pytest.raises(DatasetTooLargeError):
        adapter.load(content, narrow)


def test_excel_is_held_to_the_csv_duplicate_rule() -> None:
    """A workbook repeating a column title is refused, not silently renamed.

    Without this, pandas would turn ``a, a`` into ``a, a.1`` and the caller
    would end up modelling a column they never named.
    """
    workbook = frame_to_xlsx(pd.DataFrame([[1, 2]], columns=["a", "a"]))

    with pytest.raises(DuplicateColumnsError):
        ExcelAdapter().load(workbook, Settings())


def test_json_object_keys_cannot_collide(settings: Settings) -> None:
    """JSON gets duplicate-freedom from its own syntax, and it is checked.

    A document cannot express two identical keys in one object, and a
    flattened nested key lands in the same column rather than creating a
    second one — so the frame the adapter produces has unique names by
    construction, and the shared check confirms it rather than assuming it.
    """
    frame = JSONAdapter().load(
        json.dumps([{"x.y": 1, "x": {"y": 2}}]).encode("utf-8"), settings
    )

    assert list(frame.columns) == ["x.y"]


# ---------------------------------------------------------------------------
# 14. Binary content under a text extension
# ---------------------------------------------------------------------------


def test_a_workbook_renamed_to_csv_is_refused(settings: Settings) -> None:
    """Detection trusted the extension; the CSV adapter did not trust the bytes."""
    with pytest.raises(MalformedCSVError) as exc_info:
        CSVAdapter().load(build_xlsx(HEADER, ROWS), settings)

    assert "not text" in str(exc_info.value)


def test_a_workbook_renamed_to_json_is_refused(settings: Settings) -> None:
    """The same guard protects the JSON adapter."""
    with pytest.raises(InvalidJSONError):
        JSONAdapter().load(build_xlsx(HEADER, ROWS), settings)


def test_binary_detection_recognises_containers_and_nul_bytes() -> None:
    """Text files do not begin with a ZIP header or hold NUL bytes."""
    assert looks_binary(b"PK\x03\x04rest") is True
    assert looks_binary(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") is True
    assert looks_binary(b"a,b\n1,2\n") is False
    assert looks_binary(b"col\nva\x00lue\n") is True


# ---------------------------------------------------------------------------
# 15. Equivalence across formats — the point of the whole layer
# ---------------------------------------------------------------------------


def test_the_three_formats_produce_the_same_frame(settings: Settings) -> None:
    """The same logical data, written three ways, parses to one table."""
    csv_frame = CSVAdapter().load(build_csv(HEADER, ROWS), settings)
    xlsx_frame = ExcelAdapter().load(build_xlsx(HEADER, ROWS), settings)
    json_frame = JSONAdapter().load(build_json(HEADER, ROWS), settings)

    pd.testing.assert_frame_equal(csv_frame, xlsx_frame)
    pd.testing.assert_frame_equal(csv_frame, json_frame)


def test_equivalent_csv_and_json_fingerprint_identically(
    settings: Settings,
) -> None:
    """Identity is the data, so re-exporting a dataset does not rename it."""
    content = build_csv(HEADER, ROWS)
    csv_frame = CSVAdapter().load(content, settings)
    json_frame = JSONAdapter().load(csv_as_json(content), settings)

    assert (
        fingerprint_dataset(csv_frame).value == fingerprint_dataset(json_frame).value
    )


def test_equivalent_csv_and_excel_fingerprint_identically(
    settings: Settings,
) -> None:
    """The same holds for a spreadsheet export of the same table."""
    content = build_csv(HEADER, ROWS)
    csv_frame = CSVAdapter().load(content, settings)
    xlsx_frame = ExcelAdapter().load(csv_as_xlsx(content), settings)

    assert (
        fingerprint_dataset(csv_frame).value == fingerprint_dataset(xlsx_frame).value
    )


def test_an_enveloped_json_fingerprints_like_its_bare_array(
    settings: Settings,
) -> None:
    """The envelope is packaging, not data, so it does not change identity."""
    content = build_csv(HEADER, ROWS)
    bare = JSONAdapter().load(csv_as_json(content), settings)
    wrapped = JSONAdapter().load(csv_as_json(content, envelope="rows"), settings)

    assert fingerprint_dataset(bare).value == fingerprint_dataset(wrapped).value


def test_the_filename_does_not_affect_the_fingerprint(settings: Settings) -> None:
    """Identity is content. The same bytes under three names are one dataset."""
    service = DatasetProfilingService(settings)
    content = build_csv(HEADER, ROWS)

    fingerprints = {
        fingerprint_dataset(
            service.load_content(name, content).frame
        ).value
        for name in ("a.csv", "b.csv", "../../secret.csv")
    }

    assert len(fingerprints) == 1


def test_a_missing_value_survives_every_format(settings: Settings) -> None:
    """A blank field, an empty cell and a JSON null are the same absence."""
    frame = frame_from_csv(build_csv(["a", "b"], [[1, None], [2, 5]]))

    csv_frame = CSVAdapter().load(build_csv(["a", "b"], [[1, None], [2, 5]]), settings)
    xlsx_frame = ExcelAdapter().load(frame_to_xlsx(frame), settings)
    json_frame = JSONAdapter().load(frame_to_json(frame), settings)

    for parsed in (csv_frame, xlsx_frame, json_frame):
        assert parsed["b"].isna().tolist() == [True, False]


# ---------------------------------------------------------------------------
# 16. Values are preserved, not edited
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_format", ["csv", "xlsx", "json"])
def test_values_and_column_names_are_preserved(source_format: str) -> None:
    """Ingestion refuses or accepts. It never quietly rewrites the data."""
    settings = Settings()
    header = ["  spaced  ", "Mixed Case", "unicode_café"]
    rows: list[list[object]] = [["  keep me  ", "Value", "café"]]
    payloads = {
        "csv": (CSVAdapter(), build_csv(header, rows)),
        "xlsx": (ExcelAdapter(), build_xlsx(header, rows)),
        "json": (JSONAdapter(), build_json(header, rows)),
    }
    adapter, content = payloads[source_format]

    frame = adapter.load(content, settings)

    assert list(frame.columns) == header
    assert frame.iloc[0].tolist() == ["  keep me  ", "Value", "café"]


# ---------------------------------------------------------------------------
# 17. The service reads every format through the registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "builder", "expected"),
    [
        ("data.csv", build_csv, "csv"),
        ("data.xlsx", build_xlsx, "xlsx"),
        ("data.json", build_json, "json"),
    ],
)
def test_the_service_loads_every_format(
    filename: str, builder: object, expected: str, settings: Settings
) -> None:
    """One service method, three formats, one standardised result."""
    loaded = DatasetProfilingService(settings).load_content(
        filename, builder(HEADER, ROWS)  # type: ignore[operator]
    )

    assert loaded.source_format == expected
    assert list(loaded.frame.columns) == HEADER


def test_the_service_reads_the_media_type_off_the_upload(
    settings: Settings,
) -> None:
    """An extensionless upload is still readable when the client labels it."""
    service = DatasetProfilingService(settings)
    upload = Upload(build_json(HEADER, ROWS), "application/json")

    loaded = asyncio.run(service.load_upload(upload, "dataset"))

    assert loaded.source_format == "json"


def test_the_service_reports_the_formats_it_accepts(settings: Settings) -> None:
    """The advertised set is the intersection of settings and the registry."""
    assert DatasetProfilingService(settings).supported_formats() == (
        "csv",
        "xlsx",
        "json",
    )


def test_a_narrowed_service_accepts_only_what_it_registers() -> None:
    """Narrowing the registry narrows the API, with no other change needed."""
    service = DatasetProfilingService(
        Settings(), DatasetAdapterRegistry([CSVAdapter()])
    )

    assert service.supported_formats() == ("csv",)
    with pytest.raises(UnsupportedFileTypeError):
        service.load_content("data.xlsx", build_xlsx(HEADER, ROWS))


# ---------------------------------------------------------------------------
# 18. Errors leak nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("data.xlsx", b"not a workbook at all"),
        ("data.json", b"{ broken"),
        ("data.csv", b"a,b\n1,2\n3,4,5\n"),
    ],
)
def test_a_content_error_leaks_no_internals(
    filename: str, content: bytes, settings: Settings
) -> None:
    """No traceback, no module name, no path — in the message or the details."""
    service = DatasetProfilingService(settings)

    with pytest.raises(InvalidDatasetContentError) as exc_info:
        service.load_content(filename, content)

    error = exc_info.value
    text = f"{error.message} {json.dumps(error.details, default=str)}".lower()
    for leak in (
        "traceback",
        "openpyxl",
        "zipfile",
        "/home/",
        "site-packages",
        "\\users\\",
        "object at 0x",
    ):
        assert leak not in text


def test_every_content_error_shares_one_parent(settings: Settings) -> None:
    """A caller can catch one class for 'the content was unusable'."""
    assert issubclass(MalformedCSVError, InvalidDatasetContentError)
    assert issubclass(InvalidExcelError, InvalidDatasetContentError)
    assert issubclass(InvalidJSONError, InvalidDatasetContentError)
    assert InvalidDatasetContentError.status_code == 422


# ---------------------------------------------------------------------------
# 19. Nothing uploaded is ever executed
# ---------------------------------------------------------------------------


def test_an_excel_formula_is_never_evaluated(settings: Settings) -> None:
    """A formula cell yields the value cached in the file, and nothing else.

    ``=WEBSERVICE(...)`` is the interesting case because evaluating it would
    make the server fetch a URL the uploader chose. It does not: the workbook
    carries no cached result for a formula written programmatically, so the
    cell arrives empty. No request is made, no expression is computed, and no
    external name is resolved — a workbook is read as stored data.
    """
    workbook = build_xlsx(
        ["label", "note"],
        [["kept", '=WEBSERVICE("http://example.invalid/steal")']],
    )

    frame = ExcelAdapter().load(workbook, settings)

    assert frame.loc[0, "label"] == "kept"
    assert pd.isna(frame.loc[0, "note"])


def test_a_json_string_that_looks_like_code_is_read_as_data(
    settings: Settings,
) -> None:
    """``json.loads`` builds values, never behaviour."""
    payload = "__import__('os').system('id')"
    frame = JSONAdapter().load(
        json.dumps([{"note": payload}]).encode("utf-8"), settings
    )

    assert frame.loc[0, "note"] == payload


def test_a_csv_formula_is_read_as_data(settings: Settings) -> None:
    """The CSV path has always done this; the refactor did not change it."""
    payload = "=cmd|'/c calc'!A0"
    frame = CSVAdapter().load(build_csv(["note"], [[payload]]), settings)

    assert frame.loc[0, "note"] == payload


def test_ingestion_imports_no_execution_machinery() -> None:
    """The ingestion package reaches no shell, process, socket or eval.

    An AST-free check on purpose: what matters is that the modules do not
    *hold* these names, whether they were imported or built dynamically.
    """
    import app.services.datasets.ingestion as package
    from app.services.datasets.ingestion import (
        csv_adapter,
        detection,
        excel_adapter,
        json_adapter,
        registry,
    )

    forbidden = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "httpx",
        "eval",
        "exec",
        "compile",
        "open",
        "Path",
        "pickle",
    }
    for module in (
        package,
        csv_adapter,
        detection,
        excel_adapter,
        json_adapter,
        registry,
    ):
        leaked = forbidden & set(vars(module))
        assert not leaked, f"{module.__name__} exposes {sorted(leaked)}"


def test_no_adapter_touches_the_filesystem(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every adapter parses bytes. None of them opens a file.

    ``open`` is replaced with something that fails loudly, so a read of any
    path — including one smuggled in through a filename — is a test failure
    rather than something to be noticed later. The uploads are built first,
    because *writing* the fixture workbook legitimately uses a temporary file
    and it is the reading side that is under test.
    """
    uploads = [
        ("../../etc/passwd.csv", build_csv(HEADER, ROWS)),
        ("C:\\secret.xlsx", build_xlsx(HEADER, ROWS)),
        ("/etc/shadow.json", build_json(HEADER, ROWS)),
    ]

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"an adapter opened a file: {args!r}")

    monkeypatch.setattr(io, "open", refuse)
    monkeypatch.setattr("builtins.open", refuse)

    service = DatasetProfilingService(settings)
    for filename, content in uploads:
        assert service.load_content(filename, content).frame.shape == (3, 4)


# ---------------------------------------------------------------------------
# 20. Malicious filenames stay display text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "../../secret.xlsx",
        "..\\..\\secret.json",
        "C:\\Users\\me\\keys.csv",
        "/etc/passwd.csv",
    ],
)
def test_a_crafted_filename_is_reduced_to_a_bare_name(
    filename: str, settings: Settings
) -> None:
    """Path separators never survive ingestion, whatever the format."""
    service = DatasetProfilingService(settings)
    builders = {
        ".csv": build_csv,
        ".xlsx": build_xlsx,
        ".json": build_json,
    }
    suffix = "." + filename.rsplit(".", 1)[-1]
    loaded = service.load_content(filename, builders[suffix](HEADER, ROWS))

    assert "/" not in loaded.filename
    assert "\\" not in loaded.filename
    assert ".." not in loaded.filename


# ---------------------------------------------------------------------------
# 21. Sanity: the sample CSV still behaves exactly as before
# ---------------------------------------------------------------------------


def test_the_sample_dataset_is_unchanged_by_the_refactor(
    settings: Settings,
) -> None:
    """A regression guard on the format the project already supported."""
    frame = DatasetProfilingService(settings).load_content(
        "dataset.csv", sample_csv()
    ).frame

    assert list(frame.columns) == [
        "user_id",
        "age",
        "score",
        "city",
        "signup_date",
        "plan",
    ]
    assert frame.shape == (6, 6)


# ---------------------------------------------------------------------------
# 22. The boundary, enforced rather than described
# ---------------------------------------------------------------------------


def _imports(path: object) -> set[str]:
    """Every module name a file imports, read from its syntax tree."""
    import ast

    names: set[str] = set()
    tree = ast.parse(Path(str(path)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_only_the_dataset_service_reads_the_ingestion_package() -> None:
    """Format knowledge stays in one package, and this is what pins it there.

    If a route, the experiment runner, the agent adapter or the ML layer ever
    imports an adapter directly, "the rest of the application does not care
    about formats" stops being true — and it stops being true quietly. So the
    set of modules allowed to import it is written down, and it is short.
    """
    backend = Path(__file__).resolve().parents[1] / "app"
    allowed = {
        "services/datasets/service.py",
        "services/datasets/loader.py",
    }

    importers = {
        str(path.relative_to(backend)).replace("\\", "/")
        for path in backend.rglob("*.py")
        if "ingestion/" not in str(path.relative_to(backend)).replace("\\", "/")
        and any(
            name.startswith("app.services.datasets.ingestion")
            for name in _imports(path)
        )
    }

    assert importers == allowed


@pytest.mark.parametrize("package", ["agent", "ml", "rag", "llm"])
def test_no_other_layer_knows_a_file_format_exists(package: str) -> None:
    """``agent``, ``ml``, ``rag`` and ``llm`` import no adapter and no reader.

    They did not have to change to gain two formats, and this test is what
    keeps that true: none of them may import the ingestion package, the Excel
    reader, or FastAPI's upload type.
    """
    root = Path(__file__).resolve().parents[2] / package
    forbidden = ("app.services.datasets", "openpyxl", "fastapi", "starlette")

    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        leaked = sorted(
            name
            for name in _imports(path)
            if any(name == bad or name.startswith(bad + ".") for bad in forbidden)
        )
        assert not leaked, f"{path} imports {leaked}"


def test_the_agent_is_never_told_the_format() -> None:
    """The planner's context carries four facts, and the format is not one.

    Deliberate: a run that could branch on the file format would answer
    differently for the same data, which is the opposite of the point.
    """
    from app.services.agent.datasets import RequestDataset

    dataset = RequestDataset(
        frame=pd.DataFrame({"a": [1, 2]}),
        fingerprint="abc123",
        filename="book.xlsx",
        source_format="xlsx",
        row_count=2,
        column_count=1,
        columns=("a",),
    )

    context = dataset.planner_context()

    assert set(context) == {
        "dataset_available",
        "dataset_name",
        "dataset_rows",
        "dataset_columns",
    }
    assert "xlsx" not in json.dumps(context)
    # But the caller is told, on the response.
    assert dataset.as_dict()["source_format"] == "xlsx"
