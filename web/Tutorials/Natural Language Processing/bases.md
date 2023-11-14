## **Libraries**
### **nltk**
The Natural Language Toolkit (**``NLTK``**) is a comprehensive library for working with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, making it a valuable tool for various NLP tasks.

```py
# Install NLTK
pip install nltk

# Import NLTK
import nltk

# Download NLTK data
nltk.download('punkt')
```

### **spacy**
SpaCy is an open-source library designed specifically for Natural Language Processing. It offers pre-trained models for various languages and supports tasks like tokenization, part-of-speech tagging, named entity recognition, and more.

```py
# Install SpaCy
pip install spacy

# Download SpaCy model for English
python -m spacy download en
```

### **re (Regular Expressions)**
The **``re``** library in Python is a powerful tool for working with regular expressions. Regular expressions enable flexible and sophisticated pattern matching in text, making it easier to extract or manipulate specific information.

```py
import re

# Sample text
text = "Natural Language Processing is fascinating. NLP opens up a world of possibilities."

# Define a pattern to search for
pattern = re.compile(r'\bNLP\b')

# Search for the pattern in the text
matches = pattern.findall(text)

# Print the matches
print(matches)
```

### **gensim**
**``Gensim``** is a library for topic modeling and document similarity analysis. It is particularly useful for tasks like document similarity comparison and topic modeling using techniques like Latent Semantic Analysis.

```py
# Install Gensim
pip install gensim
```

### **fasttext**
**``FastText``** is an open-source, free, lightweight library that allows users to learn text representations and perform text classification tasks efficiently. It is an extension of the Word2Vec model.

```py
# Install FastText
pip install fasttext
```

## **Reading Text**
### **txt files**
This section provides code examples for reading and manipulating text data stored in plain text (txt) files. It covers operations such as reading, writing, and appending text, as well as working with files containing different languages, including Arabic.

```py
# Writing to a text file
%%writefile text1.txt 
bla bla bla bla bla bla bla

# Reading from a text file
my_file = open('test.txt')
content = my_file.read()
my_file.seek(0)
lines = my_file.readlines()
my_file.close()

# Appending to a text file
with open('test.txt','a+') as my_file:
    my_file.write('\nThis line is being appended to test.txt')

# Appending with magic command
%%writefile -a test.txt 
This is more text being appended to test.txt 
And another line here.

# Reading the first line from a text file
with open('test.txt','r') as txt: 
    first_line = txt.readlines()[0] 
print(first_line)

# Reading the first line with context manager
with open('test.txt','r') as txt: 
    first_line = txt.readlines()[0] 
print(first_line)
```

### **csv files**
For handling structured data, particularly in tabular form, this section demonstrates reading from and writing to CSV files. Additionally, it showcases reading data from an Excel file using the Pandas library.

```py
# Reading the first line from a CSV file
with open('test.txt','r') as txt: 
    first_line = txt.readlines()[0] 
print(first_line)

# Reading data from an Excel file
data = pd.read_excel('02.xlsx')# , skiprows = 2) 
data.head()

# Writing to a CSV file
outfile = open('03.csv', 'w') 
outfile.write('a') 
outfile.close()

# Writing to an Excel file
outfile = open('04.xls', 'w') 
outfile.write('a') 
outfile.close()

# Writing to a text file
f= open('5.txt','w') #write 
f.write('write this line in the file') 
f.close()

# Reading from a text file
f= open('5.txt','r') #read 
for a in f: 
    print(a)

# Appending to a text file
f= open('5.txt','a') #append 
f.write('\nmore lines') 
f.close()

# Writing to a CSV file using pandas
import numpy as np 
data = pd.DataFrame(pd.Series(np.random.rand(10000))) 
data.head(20) 
data.to_csv('6.csv')

# Writing to a text file with Arabic text
# to use arabic, encoding="utf8"
f= open('7.txt','w', encoding="utf8") 
f.write('سطور باللغة العربية')
f.close() 

# Appending to a text file with Arabic text
f= open('7.txt','a', encoding="utf8") 
f.write('\nثاني سطر ')
f.write('\nثالث سطر ')
f.write('\nرابع سطر ')
f.write('\nأخير سطر ')
f.close()

# Reading from a text file with Arabic text
f= open('7.txt','r', encoding="utf8") 
for a in f: 
    print(a)
```

## **Handling PDF**
**``PyPDF2``** is a Python library for reading and manipulating PDF files. This section covers basic operations like reading text from a PDF file, extracting text from specific pages, and merging PDF files.

```py
# Install PyPDF2
pip install PyPDF2

import PyPDF2

# Reading text from a PDF file
f= open('US_Declaration.pdf','rb')
pdf_reader = PyPDF2.PdfFileReader(f)

num_pages = pdf_reader.numPages

page_one = pdf_reader.getPage(0)
page_one_text = page_one.extractText()
print(page_one_text)

# Reading text from all pages of a PDF file
f = open('US_Declaration.pdf','rb')
pdf_text = [0]
pdf_reader = PyPDF2.PdfFileReader(f)
for p in range(pdf_reader.numPages):
    page = pdf_reader.getPage(p)
    pdf_text.append(page.extractText())
f.close()
print(pdf_text)

# Merging PDF files
f = open('US_Declaration.pdf','rb')
pdf_reader = PyPDF2.PdfFileReader(f)
first_page = pdf_reader.getPage(0)

pdf_writer = PyPDF2.PdfFileWriter()

pdf_writer.addPage(first_page)
```

## **Search in Text**
This section revisits the **``re``** library, demonstrating its application for searching specific patterns or keywords within text. Regular expressions are powerful tools for identifying and extracting information based on predefined patterns.

```py
# Searching for a pattern in text
import re

# Sample text
text = "Natural Language Processing is fascinating. NLP opens up a world of possibilities."

# Define a pattern to search for
pattern = re.compile(r'\bNLP\b')

# Search for the pattern in the text
matches = pattern.findall(text)

# Print the matches
print(matches)
```