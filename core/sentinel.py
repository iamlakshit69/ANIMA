class _Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"


SILENCE_MARKER    = _Sentinel("SILENCE_MARKER")
END_OF_RESPONSE   = _Sentinel("END_OF_RESPONSE")
END_OF_SPEECH     = _Sentinel("END_OF_SPEECH")
INTERRUPT         = _Sentinel("INTERRUPT")