# # TODO: create tests. This is from an old project

import os
import sqlite3

# import pytest
#
#
#
# @pytest.fixture(scope="module")
# def backup_existing_db():
#     if os.path.exists(DB_NAME):
#         os.rename(DB_NAME, DB_NAME + ".bak")
#
#
# @pytest.fixture()
# def get_db():
#     # run before each test
#     db = sqlite3.connect(DB_NAME)
#     cur = db.cursor()
#     tables = cur.execute(
#         "select name from sqlite_master where type=='table'"
#     ).fetchall()
#     # A list of tuples is returned, turn to list
#     if len(tables) > 0:
#         tables = [i[0] for i in tables]
#     yield db, cur, tables
#     db.commit()
#
#
# def test_empty_clean(backup_existing_db, get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert not clean(cur, tables)
#
#
# def test_init(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert init(cur, tables)
#
#
# def test_init_already_done(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert not init(cur, tables)
#
#
# def test_add_doc(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert add_doc("test_docs/doc1.txt", cur) == 772
#
#
# def test_add_doc_again(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert add_doc("test_docs/doc1.txt", cur) == -1
#
#
# def test_add_dir1(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert add_dir("test_docs", cur) == 6
#
#
# def test_clean2(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert clean(cur, tables)
#
#
# def test_add_dir2(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert add_dir("test_docs", cur) == 7
#
#
# def test_query_word(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert query_word("government", cur) == 42
#
#
# def test_notfound_word(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     assert query_word("gover", cur) == 0
#
#
# def test_clean_3(get_db):
#     db, cur, tables = get_db[0], get_db[1], get_db[2]
#     x = clean(cur, tables)
#     os.remove(DB_NAME)
#     if os.path.exists(DB_NAME + ".bak"):
#         os.rename(DB_NAME + ".bak", DB_NAME)
#     assert x
