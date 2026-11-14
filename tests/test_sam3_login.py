import contextlib
import io
import unittest
from unittest.mock import patch

from cvat_agent.sam3_login import clean_token, main


class Sam3LoginTests(unittest.TestCase):
    def test_clipboard_wrappers_are_removed(self):
        self.assertEqual(clean_token('  "hf_example"  '), 'hf_example')
        self.assertEqual(clean_token('Bearer hf_example'), 'hf_example')

    def test_login_does_not_print_token_or_save_git_credentials(self):
        output = io.StringIO()
        token = 'hf_fake_secret_test_value'
        with patch('cvat_agent.sam3_login.prompt_token', return_value=token), \
             patch('huggingface_hub.login') as login, contextlib.redirect_stdout(output):
            self.assertEqual(main(), 0)
        login.assert_called_once_with(token=token, add_to_git_credential=False, skip_if_logged_in=False)
        self.assertNotIn(token, output.getvalue())

    def test_cancel_does_not_attempt_login(self):
        with patch('cvat_agent.sam3_login.prompt_token', return_value=None), \
             patch('huggingface_hub.login') as login, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 1)
        login.assert_not_called()


if __name__ == '__main__':
    unittest.main()
