# File: commenting.py 
# Authors: Your Names Here
# Description: Small "Hello, World!" program to demonstrate good commenting and organization of a Python source file.

# IMPORTS
import sys

# GLOBAL VARIABLES
SUCCESS = 0
FAILURE = -1
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
MESSAGE = "Hello, World!"

# print_message: Print a message to standard output.
# Parameters:
#   - message: The string to print.
# Returns:
#   - SUCCESS (0) on success, FAILURE (-1) on failure.
def print_message(message: str) -> int:
    try:
        # Use low-level write/flush to detect potential I/O errors.
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
    except Exception:
        # Any exception while writing means failure.
        return FAILURE

    return SUCCESS

# main: Entry point of the program.
# Parameters:
#   - argv: Optional list of command-line arguments.
# Returns:
#   - EXIT_SUCCESS (0) on success, EXIT_FAILURE (1) on failure.
def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    # Print the hello world message and handle potential errors.
    if print_message(MESSAGE) != SUCCESS:
        # Print error message to stderr.
        sys.stderr.write("Error: Failed to print message.\n")
        sys.stderr.flush()
        return EXIT_FAILURE

    return EXIT_SUCCESS


if __name__ == "__main__":
    # Exit with an appropriate status code for the calling environment.
    sys.exit(main())
