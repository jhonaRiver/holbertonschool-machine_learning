#!/usr/bin/env python3
"""Script that takes in input from the user and responds."""


words = ['bye', 'goodbye', 'quit', 'exit']
while True:
    request = input("Q: ")
    if request.lower() in words:
        print('A: Goodbye')
        break
    else:
        print('A: ')
