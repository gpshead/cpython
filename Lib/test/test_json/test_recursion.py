from test import support
from test.test_json import PyTest, CTest


class JSONTestObject:
    pass


class TestRecursion:
    def test_listrecursion(self):
        x = []
        x.append(x)
        try:
            self.dumps(x)
        except ValueError as exc:
            self._assert_circular_error_notes(exc, "when serializing list item 0")
        else:
            self.fail("didn't raise ValueError on list recursion")
        x = []
        y = [x]
        x.append(y)
        try:
            self.dumps(x)
        except ValueError as exc:
            self._assert_circular_error_notes(exc, "when serializing list item 0")
        else:
            self.fail("didn't raise ValueError on alternating list recursion")
        y = []
        x = [y, y]
        # ensure that the marker is cleared
        self.dumps(x)

    def test_dictrecursion(self):
        x = {}
        x["test"] = x
        try:
            self.dumps(x)
        except ValueError as exc:
            self._assert_circular_error_notes(exc, "when serializing dict item 'test'")
        else:
            self.fail("didn't raise ValueError on dict recursion")
        x = {}
        y = {"a": x, "b": x}
        # ensure that the marker is cleared
        self.dumps(x)

    def test_defaultrecursion(self):
        class RecursiveJSONEncoder(self.json.JSONEncoder):
            recurse = False
            def default(self, o):
                if o is JSONTestObject:
                    if self.recurse:
                        return [JSONTestObject]
                    else:
                        return 'JSONTestObject'
                return self.json.JSONEncoder.default(o)

        enc = RecursiveJSONEncoder()
        self.assertEqual(enc.encode(JSONTestObject), '"JSONTestObject"')
        enc.recurse = True
        try:
            with support.infinite_recursion(5000):
                enc.encode(JSONTestObject)
        except ValueError as exc:
            notes = exc.__notes__
            # Should have reasonable number of notes and contain expected context
            self.assertLessEqual(len(notes), 10)
            self.assertGreater(len(notes), 0)
            note_strs = [str(note) for note in notes if not str(note).startswith("... (truncated")]
            self.assertTrue(any("when serializing list item 0" in note for note in note_strs))
            self.assertTrue(any("when serializing type object" in note for note in note_strs))
        else:
            self.fail("didn't raise ValueError on default recursion")


    @support.skip_emscripten_stack_overflow()
    @support.skip_wasi_stack_overflow()
    def test_highly_nested_objects_decoding(self):
        very_deep = 200000
        # test that loading highly-nested objects doesn't segfault when C
        # accelerations are used. See #12017
        with self.assertRaises(RecursionError):
            with support.infinite_recursion():
                self.loads('{"a":' * very_deep + '1' + '}' * very_deep)
        with self.assertRaises(RecursionError):
            with support.infinite_recursion():
                self.loads('{"a":' * very_deep + '[1]' + '}' * very_deep)
        with self.assertRaises(RecursionError):
            with support.infinite_recursion():
                self.loads('[' * very_deep + '1' + ']' * very_deep)

    @support.skip_wasi_stack_overflow()
    @support.skip_emscripten_stack_overflow()
    @support.requires_resource('cpu')
    def test_highly_nested_objects_encoding(self):
        # See #12051
        l, d = [], {}
        for x in range(200_000):
            l, d = [l], {'k':d}
        with self.assertRaises(RecursionError):
            with support.infinite_recursion(5000):
                self.dumps(l, check_circular=False)
        with self.assertRaises(RecursionError):
            with support.infinite_recursion(5000):
                self.dumps(d, check_circular=False)

    @support.skip_emscripten_stack_overflow()
    @support.skip_wasi_stack_overflow()
    def test_endless_recursion(self):
        # See #12051
        class EndlessJSONEncoder(self.json.JSONEncoder):
            def default(self, o):
                """If check_circular is False, this will keep adding another list."""
                return [o]

        with self.assertRaises(RecursionError):
            with support.infinite_recursion(1000):
                EndlessJSONEncoder(check_circular=False).encode(5j)

    def test_circular_reference_error_notes(self):
        """Test that circular reference errors have reasonable exception notes."""
        # Test simple circular list
        x = []
        x.append(x)
        try:
            self.dumps(x, check_circular=True)
        except ValueError as exc:
            self._assert_circular_error_notes(exc, "when serializing list item 0")
        else:
            self.fail("didn't raise ValueError on list recursion")

        # Test simple circular dict
        y = {}
        y['self'] = y
        try:
            self.dumps(y, check_circular=True)
        except ValueError as exc:
            self._assert_circular_error_notes(exc, "when serializing dict item 'self'")
        else:
            self.fail("didn't raise ValueError on dict recursion")

    def test_nested_circular_reference_notes(self):
        """Test that nested circular reference notes don't contain duplicates."""
        # Create a nested circular reference
        z = []
        nested = {'deep': [z]}
        z.append(nested)

        try:
            self.dumps(z, check_circular=True)
        except ValueError as exc:
            notes = getattr(exc, '__notes__', [])
            # All non-truncation notes should be unique
            actual_notes = [note for note in notes if not str(note).startswith("... (truncated")]
            unique_notes = list(dict.fromkeys(actual_notes))  # preserves order, removes duplicates
            self.assertEqual(len(actual_notes), len(unique_notes),
                           f"Found duplicate notes: {actual_notes}")
        else:
            self.fail("didn't raise ValueError on nested circular reference")

    def test_recursion_error_when_check_circular_false(self):
        """Test that RecursionError is raised when check_circular=False."""
        x = []
        x.append(x)

        with self.assertRaises(RecursionError):
            with support.infinite_recursion(1000):
                self.dumps(x, check_circular=False)

    def test_deep_recursion_note_handling(self):
        """Test that deep recursion scenarios don't create excessive duplicate notes."""
        # Create a scenario that triggers deep recursion through custom default method
        class DeepObject:
            def __init__(self, value):
                self.value = value

        class DeepEncoder(self.json.JSONEncoder):
            def default(self, o):
                if isinstance(o, DeepObject):
                    return [DeepObject(o.value + 1)] if o.value < 10 else "end"
                return super().default(o)

        encoder = DeepEncoder(check_circular=True)

        try:
            encoder.encode(DeepObject(0))
        except (ValueError, RecursionError) as exc:
            notes = getattr(exc, '__notes__', [])

            # Should have reasonable number of notes without excessive duplication
            self.assertLessEqual(len(notes), 20)

            # Count occurrences of each note to verify no excessive duplication
            note_counts = {}
            for note in notes:
                note_str = str(note)
                if not note_str.startswith("... (truncated"):
                    note_counts[note_str] = note_counts.get(note_str, 0) + 1

            # No note should appear excessively
            max_count = max(note_counts.values()) if note_counts else 0
            self.assertLessEqual(max_count, 5,
                               f"Found excessive duplicate notes: {note_counts}")

    def _assert_circular_error_notes(self, exc, expected_context):
        """Helper method to assert circular reference error notes are reasonable."""
        notes = getattr(exc, '__notes__', [])

        # Should have reasonable number of notes (not thousands)
        self.assertLessEqual(len(notes), 10)
        self.assertGreater(len(notes), 0)

        # Should contain expected context
        self.assertTrue(any(expected_context in str(note) for note in notes),
                       f"Expected context '{expected_context}' not found in notes: {notes}")


class TestPyRecursion(TestRecursion, PyTest): pass
class TestCRecursion(TestRecursion, CTest): pass
