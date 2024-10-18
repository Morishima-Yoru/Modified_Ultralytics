
    
NO_COMPUTATIONAL_IMPLEMENTED = \
    NotImplementedError("Attributes not initialized properly. You need to implement your computational method")
NO_BUILD = \
    NotImplementedError("Wrapper does not build properly. Call build() first.")
CHANNEL_MISMATCH_NOTALLOW = \
    lambda c1, c2: ValueError(f"This Stage Wrapper doesn't allow different input/output channels. Got c1: {c1}, c2: {c2}")