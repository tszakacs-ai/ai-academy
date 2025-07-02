import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from anonymize_mails import anonymize_text, anonymize_documents

class TestAnonymization(unittest.TestCase):
    """Test the document anonymization functionality"""

    def test_anonymize_text(self):
        """Test that sensitive information is properly anonymized in text"""
        # Sample text with different types of sensitive information
        sample_text = """
        From: John Smith <john.smith@example.com>
        To: support@acme.corp
        Subject: Invoice Payment Details

        Dear Support Team,

        Please process the payment for invoice #12345 to the following account:
        IBAN: IT60X0542811101000000123456
        Account Holder: Acme Corporation
        
        You can reach me at my mobile number +39 333 1234567 or landline 02 12345678.
        My fiscal code is SMTJHN80A01H501U.
        
        Our company is located in Milan, Italy.
        
        Best regards,
        John Smith
        """
        
        # Anonymize the text
        anonymized = anonymize_text(sample_text)
        
        # Check that sensitive information has been anonymized
        self.assertNotIn("john.smith@example.com", anonymized)
        self.assertNotIn("IT60X0542811101000000123456", anonymized)
        self.assertIn("[IT60*****************]", anonymized)
        self.assertNotIn("John Smith", anonymized)
        self.assertIn("[Nome_", anonymized)  # Names are anonymized with a counter
        self.assertNotIn("SMTJHN80A01H501U", anonymized)
        self.assertIn("[SMT***********]", anonymized)
        self.assertNotIn("+39 333 1234567", anonymized)
        self.assertIn("[+39***********]", anonymized)
        self.assertNotIn("Milan", anonymized)
        self.assertIn("[Luogo_", anonymized)
        self.assertNotIn("Acme Corporation", anonymized)
        self.assertIn("[Azienda_", anonymized)

    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('builtins.open')
    def test_anonymize_documents(self, mock_open, mock_isdir, mock_listdir):
        """Test the document anonymization process"""
        # Setup mocks
        mock_listdir.return_value = ['doc1.txt', 'doc2.txt', 'subdir']
        mock_isdir.side_effect = lambda path: path.endswith('subdir')
        
        # Setup file mock to return sample text
        file_mock = MagicMock()
        file_mock.__enter__.return_value.read.return_value = "Hello from John Doe (john.doe@example.com)"
        mock_open.return_value = file_mock
        
        # Call the function
        result = anonymize_documents('input_dir', 'output_dir')
        
        # Verify that files were processed
        self.assertEqual(len(result), 2)
        
        # Verify that the write operation contains anonymized content
        write_calls = [call[0][0] for call in file_mock.__enter__.return_value.write.call_args_list]
        self.assertTrue(any('[Nome_' in call for call in write_calls))
        self.assertTrue(any('[Email_' in call for call in write_calls))

if __name__ == "__main__":
    unittest.main()
