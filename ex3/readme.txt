Ex3 - api changes and reasonings:

changed "get_available_device" function to support macOs GPU accelaration, not just cuda.
    - added an import to "platform" library, and the helper function "_running_on_mac()".

