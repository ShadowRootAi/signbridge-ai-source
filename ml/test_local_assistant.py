"""Contract tests for local correction memory without invoking Ollama."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from local_assistant import LocalAssistant


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'ραντεβού Τρίτη'
TARGET = 'ραντεβού Πέμπτη στις έξι'


class LocalAssistantTests(unittest.TestCase):
    def setUp(self):
        runtime = (ROOT / '.runtime-tmp').resolve()
        self.workspace = TemporaryDirectory(prefix='assistant-tests-', dir=runtime)
        self.addCleanup(self.workspace.cleanup)
        self.database = Path(self.workspace.name) / 'memory.sqlite3'
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return 'Απάντηση δοκιμής'

    def assistant(self, database=None):
        return LocalAssistant(database or self.database, chat=self.chat)

    def prompt_text(self):
        return '\n'.join(message['content'] for message in self.calls[-1])

    def remember(self, assistant, target=TARGET):
        return assistant.respond({'message': 'Κράτησε τη διόρθωση.',
                                  'correction': {'from': SOURCE, 'to': target}})

    def test_explicit_correction_is_used_after_reopening(self):
        first = self.assistant()
        saved = self.remember(first)
        self.assertTrue(saved['remembered'])
        self.assertEqual(self.calls, [], 'Storing an explicit correction must not invoke the language model')
        del first

        reopened = self.assistant()
        reply = reopened.respond({'message': 'Τι ισχύει για το «ραντεβού τρίτη»;'})
        self.assertEqual(reply['reply'], 'Απάντηση δοκιμής')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(reply['appliedCorrections'], [{'from': SOURCE, 'to': TARGET}])
        self.assertIn(TARGET, self.prompt_text())

    def test_conversation_text_is_not_implicitly_learned(self):
        self.assistant().respond({'message': f'Εννοούσα {TARGET} όταν είπα {SOURCE}.'})
        reply = self.assistant().respond({'message': SOURCE})
        self.assertEqual(reply['appliedCorrections'], [])
        self.assertNotIn(TARGET, self.prompt_text())

    def test_correction_does_not_match_inside_a_longer_word(self):
        self.remember(self.assistant())
        reply = self.assistant().respond({'message': 'βρες το ραντεβού Τρίτης'})
        self.assertEqual(reply['appliedCorrections'], [])
        self.assertNotIn(TARGET, self.prompt_text())

    def test_new_correction_replaces_previous_value(self):
        first = self.assistant()
        self.remember(first)
        new_target = 'ραντεβού Παρασκευή στις τρεις'
        self.remember(first, new_target)
        reply = self.assistant().respond({'message': SOURCE})
        self.assertEqual(reply['appliedCorrections'], [{'from': SOURCE, 'to': new_target}])
        self.assertIn(new_target, self.prompt_text())
        self.assertNotIn(TARGET, self.prompt_text())

    def test_memory_stays_in_the_selected_database(self):
        self.remember(self.assistant())
        separate = self.assistant(self.database.with_name('other.sqlite3'))
        reply = separate.respond({'message': SOURCE})
        self.assertEqual(reply['appliedCorrections'], [])
        self.assertNotIn(TARGET, self.prompt_text())

    def test_clearing_memory_persists_after_reopening(self):
        first = self.assistant()
        self.remember(first)
        first.respond({'memory': 'clear'})
        self.assertEqual(self.calls, [])
        reply = self.assistant().respond({'message': SOURCE})
        self.assertEqual(reply['appliedCorrections'], [])
        self.assertNotIn(TARGET, self.prompt_text())

    def test_longer_matching_correction_wins_on_the_same_words(self):
        assistant = self.assistant()
        self.remember(assistant)
        short_target = 'συνάντηση χωρίς ημερομηνία'
        assistant.respond({'message': 'Κράτησε τη δεύτερη διόρθωση.',
                           'correction': {'from': 'ραντεβού', 'to': short_target}})
        reply = assistant.respond({'message': SOURCE})
        self.assertEqual(reply['appliedCorrections'], [{'from': SOURCE, 'to': TARGET}])
        self.assertNotIn(short_target, self.prompt_text())
        reply = assistant.respond({'message': 'Έχουμε ραντεβού αύριο;'})
        self.assertEqual(reply['appliedCorrections'], [{'from': 'ραντεβού', 'to': short_target}])

    def test_rejected_recognition_does_not_pass_the_candidate_label(self):
        assistant = self.assistant()
        label = 'ΜΟΝΑΔΙΚΟ ΣΗΜΑ'
        assistant.respond({'message': 'Τι κατάλαβες;', 'recognition': {'label': label, 'accepted': False}})
        self.assertNotIn(label, self.prompt_text())
        assistant.respond({'message': 'Τι κατάλαβες;', 'recognition': {'label': label, 'accepted': True}})
        self.assertIn(label, self.prompt_text())

    def test_invalid_payload_never_reaches_the_model(self):
        assistant = self.assistant()
        cases = (
            {'message': 12},
            {'message': 'κείμενο', 'correction': {'from': SOURCE, 'to': 25}},
            {'message': 'κείμενο', 'recognition': {'accepted': 'false', 'label': SOURCE}},
        )
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                assistant.respond(payload)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
