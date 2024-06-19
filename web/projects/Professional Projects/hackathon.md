---
comments: true
---

## **Level 1: Beginner**

???+ Note "Challenge 1: Count Vowels in a String"

    - **Description:** Write a function that counts the number of vowels (a, e, i, o, u) in a given string.

    ```py
    def count_vowels(s):
        vowels = "aeiouAEIOU"
        count = 0
        # Missing loop to iterate over each character in the string
        # Missing condition to check if the character is a vowel
        # Missing increment of the count if the character is a vowel
        return count

    # Example usage:
    print(count_vowels("hello world"))  # Should return 3
    print(count_vowels("Python Programming"))  # Should return 4
    ```

    - **Instructions for Students:** 
        Complete the count_vowels function by filling in the missing parts. Specifically, you need to:
        - Loop through each character in the string ``s``.
        - Check if the character is a vowel (present in the ``vowels`` string).
        - Increment the ``count`` if the character is a vowel.
    
    - **Example of Missing Parts (for reference):**
        ```py
        for char in s:
            if char in vowels:
                count += 1
        ```

???+ Note "Challenge 2: Sum of Even Numbers"

    - **Description:** Write a function that calculates the sum of all even numbers from 1 to a given number ``n``.

    ```py
    def sum_of_evens(n):
        total = 0
        # Missing loop to iterate through even numbers from 2 to n
        # Missing line to add the current even number to the total
        return total

    # Example usage:
    print(sum_of_evens(10))  # Should return 30 (2 + 4 + 6 + 8 + 10)
    print(sum_of_evens(15))  # Should return 56 (2 + 4 + 6 + 8 + 10 + 12 + 14)

    ```

    - **Instructions for Students:** 
        Complete the ``sum_of_evens`` function by filling in the missing parts. Specifically, you need to:
        - Create a loop that iterates through even numbers from 2 to ``n`` (inclusive).
        - Add each even number to the ``total``.
    
    - **Example of Missing Parts (for reference):**
        ```py
        for i in range(2, n + 1, 2):
            total += i
        ```

## **Level 2: Intermediate**

???+ Note "Challenge 1: Fibonacci Sequence"

    - **Description:** Write a function that returns the first ``n`` numbers in the Fibonacci sequence.

    ```py
    def fibonacci(n):
        fib_sequence = [0, 1]
        # Missing loop to generate the Fibonacci sequence
        # Missing line to append the sum of the last two numbers to the sequence
        return fib_sequence[:n]

    # Example usage:
    print(fibonacci(10))  # Should return [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    print(fibonacci(5))   # Should return [0, 1, 1, 2, 3]
    ```

    - **Instructions for Students:** 
        Complete the ``fibonacci`` function by filling in the missing parts. Specifically, you need to:
        - Create a loop that generates the Fibonacci sequence starting from the third element up to the ``n``-th element.
        - Append the sum of the last two elements to the ``fib_sequence``.
    
    - **Example of Missing Parts (for reference):**
        ```py
        for i in range(2, n):
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        ```


???+ Note "Challenge 2: Prime Number Checker"

    - **Description:** Write a function that checks if a given number is a prime number.

    ```py
    def is_prime(n):
        if n <= 1:
            return False
        # Missing loop to check for factors of n
        # Missing condition to check if n is divisible by i
        return True

    # Example usage:
    print(is_prime(11))  # Should return True
    print(is_prime(10))  # Should return False
    ```

    - **Instructions for Students:** 
        Complete the ``is_prime`` function by filling in the missing parts. Specifically, you need to:
        - Create a loop that generates the Fibonacci sequence starting from the third element up to the ``n``-th element.
        - Add a condition to check if ``n`` is divisible by the current value of the loop variable ``i``. If it is, return ``False``.
    
    - **Example of Missing Parts (for reference):**
        ```py
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        ```

???+ Note "Challenge 3: Longest Substring Without Repeating Characters"

    - **Description:** Write a function that finds the length of the longest substring without repeating characters in a given string.

    ```py
    def length_of_longest_substring(s):
        char_map = {}
        start = max_length = 0
        for end, char in enumerate(s):
            # Missing line 1
                start = char_map[char] + 1
            char_map[char] = end
            # Missing line 2
        return max_length

    # Example usage:
    print(length_of_longest_substring("abcabcbb"))  # Should return 3 ("abc")
    print(length_of_longest_substring("bbbbb"))     # Should return 1 ("b")
    print(length_of_longest_substring("pwwkew"))    # Should return 3 ("wke")
    ```

    - **Instructions for Students:** 
        - **Missing line 1:** Implement the condition to check if the current character ``char`` is already in ``char_map`` and its index ``char_map[char]`` is greater than or equal to ``start``.
        - **Missing line 2:** Calculate and update ``max_length`` to track the length of the longest substring without repeating characters.
    
    - **Example of Missing Parts (for reference):**
        ```py
        if char in char_map and char_map[char] >= start:
           
        
        max_length = max(max_length, end - start + 1)
        ```


