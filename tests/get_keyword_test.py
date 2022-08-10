import pytest

def test_get_keyword(auto_tag_model,lyric_data):
    for lyric in lyric_data:
        keyword = auto_tag_model.get_keyword(lyric)
        assert keyword
        print(keyword)
    