#!/bin/bash

# SimpleLang Compiler Demo Script
# This script demonstrates how to compile and run SimpleLang programs

echo "==================================="
echo "SimpleLang Compiler Demonstration"
echo "==================================="
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3.6 or higher."
    exit 1
fi

echo "Python version:"
python3 --version
echo

# Run the test program
echo "1. Running the test program (test_input.txt):"
echo "---------------------------------------------"
python3 simplelang_compiler.py test_input.txt
echo

# Create and run a simple program
echo "2. Creating and running a simple program:"
echo "-----------------------------------------"
cat > simple_example.sl << 'EOF'
// A simple program that calculates the sum of numbers 1 to 10
int main() {
    int sum = 0;
    for (int i = 1; i <= 10; i = i + 1) {
        sum = sum + i;
    }
    return sum;  // Should return 55
}
EOF

echo "Program content:"
cat simple_example.sl
echo
echo "Running the program:"
python3 simplelang_compiler.py simple_example.sl
echo

# Create and run a factorial program
echo "3. Creating and running a factorial program:"
echo "--------------------------------------------"
cat > factorial_example.sl << 'EOF'
// Calculate factorial of 6
int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    return factorial(6);  // Should return 720
}
EOF

echo "Program content:"
cat factorial_example.sl
echo
echo "Running the program:"
python3 simplelang_compiler.py factorial_example.sl
echo

# Clean up temporary files
rm -f simple_example.sl factorial_example.sl

echo "==================================="
echo "Demo completed successfully!"
echo "==================================="