import inspect
import quadcopter

# Get all members of the package
members = inspect.getmembers(quadcopter)

# Filter functions, classes, etc.
for name, obj in members:
    if inspect.isfunction(obj):
        print(f"Function: {name}")
    elif inspect.isclass(obj):
        print(f"Class: {name}")