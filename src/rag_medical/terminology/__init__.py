"""医学术语规范化工具。

该包只提供词典读取、文本规范化和离线候选维护能力；正式词典位于
``resources/terminology``，运行时不会被自动修改。
"""

from rag_medical.terminology.dictionary import TermDictionary, load_dictionary

__all__ = ["TermDictionary", "load_dictionary"]
