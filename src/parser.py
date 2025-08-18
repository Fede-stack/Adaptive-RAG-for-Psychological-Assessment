import xml.etree.ElementTree as ET
from typing import List, Optional
from pathlib import Path


class XMLDocumentParser:
    """
    A robust XML parser that handles malformed XML files with recovery capabilities.
    
    This parser attempts to extract valid XML elements from potentially corrupted
    files by processing them line by line and filtering out invalid content.
    """
    
    def __init__(self, min_word_count: int = 2):
        """
        Initialize the XML parser.
        
        Args:
            min_word_count (int): Minimum number of words required for a document
                                to be included in the filtered results.
        """
        self.min_word_count = min_word_count
        self._last_error: Optional[Exception] = None
    
    def parse_xml_with_recovery(self, file_path: str) -> List[str]:
        """
        Parse XML file with recovery mechanism for malformed content.
        
        This method maintains the original functionality while being part of the class.
        
        Args:
            file_path (str): Path to the XML file to parse
            
        Returns:
            List[str]: List of extracted text content from TEXT elements
        """
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                xml_string = ''
                
            for line in lines:
                try:
                    ET.fromstring('<root>' + line.strip() + '</root>')
                    xml_string += line
                except ET.ParseError:
                    continue
                    
            root = ET.fromstring('<root>' + xml_string + '</root>')
            texts = [text_elem.text for text_elem in root.findall('.//TEXT')]
            
        except Exception as e:
            self._last_error = e
            print(f"Errore generale: {e}")
            
        return texts
    
    def filter_documents(self, docss: List[List[str]]) -> List[List[str]]:
        """
        Filter documents based on minimum word count requirement. We used 2 as default. 
        
        Args:
            docss (List[List[str]]): Nested list of document collections
            
        Returns:
            List[List[str]]: Filtered document collections containing only
                           documents with sufficient word count
        """
        filtered_docss = []
        
        for doc_list in docss:
            filtered_list = [
                doc for doc in doc_list 
                if len(doc.split()) >= self.min_word_count
            ]
            if filtered_list:
                filtered_docss.append(filtered_list)
                
        return filtered_docss
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Complete processing pipeline: parse XML and return filtered texts.
        
        Args:
            file_path (str): Path to the XML file
            
        Returns:
            List[str]: Filtered text content
        """
        texts = self.parse_xml_with_recovery(file_path)
        if texts:
            # Wrap in nested list format expected by filter_documents
            filtered = self.filter_documents([texts])
            return filtered[0] if filtered else []
        return []
    
    def process_multiple_files(self, file_paths: List[str]) -> List[List[str]]:
        """
        Process multiple XML files and return filtered results.
        
        Args:
            file_paths (List[str]): List of file paths to process
            
        Returns:
            List[List[str]]: Filtered document collections from all files
        """
        all_documents = []
        
        for file_path in file_paths:
            texts = self.parse_xml_with_recovery(file_path)
            if texts:
                all_documents.append(texts)
        
        return self.filter_documents(all_documents)
    
    @property
    def last_error(self) -> Optional[Exception]:
        """get the last error that occurred during parsing."""
        return self._last_error
    
    def set_min_word_count(self, count: int) -> None:
        """update the minimum word count requirement."""
        if count < 0:
            raise ValueError("Minimum word count must be non-negative")
        self.min_word_count = count

