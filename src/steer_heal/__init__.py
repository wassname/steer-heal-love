import os

if os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package

    beartype_this_package()
