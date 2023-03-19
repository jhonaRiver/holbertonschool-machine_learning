#!/usr/bin/env python3
"""Module update_topics."""


def update_topics(mongo_collection, name, topics):
    """
    Change all topics of a school document based on the name.

    Args:
        mongo_collection (obj): pymongo collection object
        name (str): school name to update
        topics (list): list of topics approached in the school
    """
    mongo_collection.update_many({'name': name}, {'$set': {'topics': topics}})
