#!/usr/bin/env python3
"""Module list_all."""


def list_all(mongo_collection):
    """
    List all documents in a collection.

    Args:
        mongo_collection (obj): pymongo collection object
    """
    all_docs = []
    collection = mongo_collection.find()
    for document in collection:
        all_docs.append(document)
    return all_docs
