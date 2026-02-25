"""Utilities for writing StencilGroups out to a C header file."""

import itertools
import typing
import math

import _stencils


def _dump_footer(
    groups: dict[str, _stencils.StencilGroup], symbols: dict[str, int]
) -> typing.Iterator[str]:
    symbol_mask_size = max(math.ceil(len(symbols) / 32), 1)
    yield f'static_assert(SYMBOL_MASK_WORDS >= {symbol_mask_size}, "SYMBOL_MASK_WORDS too small");'
    yield ""
    yield "typedef struct {"
    yield "    void (*emit)("
    yield "        unsigned char *code, unsigned char *cold_code, unsigned char *data,"
    yield "        _PyExecutorObject *executor,"
    yield "        const _PyUOpInstruction *instruction, jit_state *state);"
    yield "    size_t hot_code_size;"
    yield "    size_t cold_code_size;"
    yield "    size_t data_size;"
    yield "    symbol_mask trampoline_mask;"
    yield "    symbol_mask got_mask;"
    yield "} StencilGroup;"
    yield ""
    yield f"static const StencilGroup shim = {groups['shim'].as_c('shim')};"
    yield ""
    yield "static const StencilGroup stencil_groups[MAX_UOP_REGS_ID + 1] = {"
    for opname, group in sorted(groups.items()):
        if opname == "shim":
            continue
        yield f"    [{opname}] = {group.as_c(opname)},"
    yield "};"
    yield ""
    yield f"static const void * const symbols_map[{max(len(symbols), 1)}] = {{"
    if symbols:
        for symbol, ordinal in symbols.items():
            yield f"    [{ordinal}] = &{symbol},"
    else:
        yield "    0"
    yield "};"


def _dump_stencil(opname: str, group: _stencils.StencilGroup) -> typing.Iterator[str]:
    cold_offset = group.cold_offset
    yield "void"
    yield f"emit_{opname}("
    yield "    unsigned char *code, unsigned char *cold_code, unsigned char *data,"
    yield "    _PyExecutorObject *executor,"
    yield "    const _PyUOpInstruction *instruction, jit_state *state)"
    yield "{"
    # Emit hot code body:
    for line in group.code.disassembly:
        yield f"    // {line}"
    if cold_offset:
        hot_body = group.code.body[:cold_offset]
        cold_body = group.code.body[cold_offset:]
    else:
        hot_body = group.code.body
        cold_body = bytearray()
    # Emit hot code body array:
    hot_stripped = hot_body.rstrip(b"\x00")
    if hot_stripped:
        yield f"    const unsigned char code_body[{len(hot_body)}] = {{"
        for i in range(0, len(hot_stripped), 8):
            row = " ".join(f"{byte:#04x}," for byte in hot_stripped[i : i + 8])
            yield f"        {row}"
        yield "    };"
    # Emit cold code body array:
    cold_stripped = cold_body.rstrip(b"\x00")
    if cold_stripped:
        yield f"    const unsigned char cold_code_body[{len(cold_body)}] = {{"
        for i in range(0, len(cold_stripped), 8):
            row = " ".join(f"{byte:#04x}," for byte in cold_stripped[i : i + 8])
            yield f"        {row}"
        yield "    };"
    # Emit data body array:
    for line in group.data.disassembly:
        yield f"    // {line}"
    data_stripped = group.data.body.rstrip(b"\x00")
    if data_stripped:
        yield f"    const unsigned char data_body[{len(group.data.body)}] = {{"
        for i in range(0, len(data_stripped), 8):
            row = " ".join(f"{byte:#04x}," for byte in data_stripped[i : i + 8])
            yield f"        {row}"
        yield "    };"
    # Data is written first (so relaxations in the code work properly):
    if data_stripped:
        yield "    memcpy(data, data_body, sizeof(data_body));"
    # Emit data holes:
    skip = False
    group.data.holes.sort(key=lambda hole: hole.offset)
    for hole, pair in itertools.zip_longest(group.data.holes, group.data.holes[1:]):
        if skip:
            skip = False
            continue
        if pair and (folded := hole.fold(pair, group.data.body)):
            skip = True
            hole = folded
        yield f"    {hole.as_c('data')}"
    # Copy hot code:
    if hot_stripped:
        yield "    memcpy(code, code_body, sizeof(code_body));"
    # Copy cold code:
    if cold_stripped:
        yield "    memcpy(cold_code, cold_code_body, sizeof(cold_code_body));"
    # Emit code holes, adjusting location for cold holes:
    skip = False
    group.code.holes.sort(key=lambda hole: hole.offset)
    for hole, pair in itertools.zip_longest(group.code.holes, group.code.holes[1:]):
        if skip:
            skip = False
            continue
        if pair and (folded := hole.fold(pair, group.code.body)):
            skip = True
            hole = folded
        if cold_offset and hole.offset >= cold_offset:
            # This hole is in cold code — adjust offset relative to cold_code
            adjusted = hole.replace(offset=hole.offset - cold_offset)
            yield f"    {adjusted.as_c('cold_code')}"
        else:
            yield f"    {hole.as_c('code')}"
    yield "}"
    yield ""


def dump(
    groups: dict[str, _stencils.StencilGroup], symbols: dict[str, int]
) -> typing.Iterator[str]:
    """Yield a JIT compiler line-by-line as a C header file."""
    for opname, group in groups.items():
        yield from _dump_stencil(opname, group)
    yield from _dump_footer(groups, symbols)
