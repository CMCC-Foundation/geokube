"""Filename <-> format-pattern parsing.

``reverse_format`` (with its helpers ``_get_parts_of_format_string`` and
``_validate_format_spec``) is vendored, with a minor adaptation, from the
``intake`` project (``intake/source/utils.py``). It is the only thing geokube
used from intake, so vendoring it lets us drop the heavy ``intake`` runtime
dependency.

Original copyright::

    Copyright (c) 2012 - 2018, Anaconda, Inc. and Intake contributors
    All rights reserved. Licensed under the BSD 2-Clause License.

The only change from upstream is replacing fsspec's ``make_path_posix`` with a
small standard-library separator normalization (geokube operates on POSIX
paths, so this is behavior-preserving for its use).
"""
from datetime import datetime
from string import Formatter

__all__ = ["reverse_format"]


def _make_path_posix(path):
    # Upstream used ``fsspec.implementations.local.make_path_posix`` purely to
    # normalize separators; geokube runs on POSIX paths, so normalizing any
    # backslashes to forward slashes is sufficient and dependency-free.
    return str(path).replace("\\", "/")


def _validate_format_spec(format_spec):
    if format_spec[-1].isalpha():
        format_spec = format_spec[:-1]
    if not format_spec.isdigit():
        raise ValueError("Format specifier must have a set width")
    return int(format_spec)


def _get_parts_of_format_string(resolved_string, literal_texts, format_specs):
    """
    Inner function of reverse_format, returns the resolved value for each
    field in pattern.
    """
    _text = resolved_string
    bits = []

    if literal_texts[-1] != "" and _text.endswith(literal_texts[-1]):
        _text = _text[: -len(literal_texts[-1])]
        literal_texts = literal_texts[:-1]
        format_specs = format_specs[:-1]

    for i, literal_text in enumerate(literal_texts):
        if literal_text != "":
            if literal_text not in _text:
                raise ValueError(
                    (
                        "Resolved string must match pattern. "
                        "'{}' not found.".format(literal_text)
                    )
                )
            bit, _text = _text.split(literal_text, 1)
            if bit:
                bits.append(bit)
        elif i == 0:
            continue
        else:
            try:
                format_spec = _validate_format_spec(format_specs[i - 1])
                bits.append(_text[0:format_spec])
                _text = _text[format_spec:]
            except Exception:
                if i == len(format_specs) - 1:
                    format_spec = _validate_format_spec(format_specs[i])
                    bits.append(_text[:-format_spec])
                    bits.append(_text[-format_spec:])
                    _text = []
                else:
                    _validate_format_spec(format_specs[i - 1])
    if _text:
        bits.append(_text)
    if len(bits) > len([fs for fs in format_specs if fs is not None]):
        bits = bits[1:]
    return bits


def reverse_format(format_string, resolved_string):
    """
    Reverse the string method format.

    Given format_string and resolved_string, find arguments that would
    give ``format_string.format(**arguments) == resolved_string``

    Parameters
    ----------
    format_string : str
        Format template string as used with str.format method
    resolved_string : str
        String with same pattern as format_string but with fields
        filled out.

    Returns
    -------
    args : dict
        Dict of the form {field_name: value} such that
        ``format_string.(**args) == resolved_string``

    Examples
    --------

    >>> reverse_format('data_{year}_{month}_{day}.csv', 'data_2014_01_03.csv')
    {'year': '2014', 'month': '01', 'day': '03'}
    >>> reverse_format('data_{year:d}_{month:d}_{day:d}.csv', 'data_2014_01_03.csv')
    {'year': 2014, 'month': 1, 'day': 3}
    >>> reverse_format('data_{date:%Y_%m_%d}.csv', 'data_2016_10_01.csv')
    {'date': datetime.datetime(2016, 10, 1, 0, 0)}
    >>> reverse_format('{state:2}{zip:5}', 'PA19104')
    {'state': 'PA', 'zip': '19104'}

    See also
    --------
    str.format : method that this reverses
    """
    fmt = Formatter()
    args = {}

    # ensure that format_string uses POSIX separators
    format_string = _make_path_posix(format_string)

    # split the string into bits
    literal_texts, field_names, format_specs, conversions = zip(
        *fmt.parse(format_string)
    )
    if not any(field_names):
        return {}

    for i, conversion in enumerate(conversions):
        if conversion:
            raise ValueError(
                ("Conversion not allowed. Found on {}.".format(field_names[i]))
            )

    # ensure that resolved string uses POSIX separators
    resolved_string = _make_path_posix(resolved_string)

    # get a list of the parts that matter
    bits = _get_parts_of_format_string(
        resolved_string, literal_texts, format_specs
    )

    for i, (field_name, format_spec) in enumerate(zip(field_names, format_specs)):
        if field_name:
            try:
                if format_spec.startswith("%"):
                    args[field_name] = datetime.strptime(bits[i], format_spec)
                elif format_spec[-1] in list("bcdoxX"):
                    args[field_name] = int(bits[i])
                elif format_spec[-1] in list("eEfFgGn"):
                    args[field_name] = float(bits[i])
                elif format_spec[-1] == "%":
                    args[field_name] = float(bits[i][:-1]) / 100
                else:
                    args[field_name] = fmt.format_field(bits[i], format_spec)
            except Exception:
                args[field_name] = bits[i]

    return args
