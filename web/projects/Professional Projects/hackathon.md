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





