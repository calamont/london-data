London Data
===========

Using Python and Beautiful Soup to scrape data about advertised rental
properties in London. Data on over 27,000 listings with 38 data points
each was obtained. Scikit-learn was used to analyse the data and to
develop classifiers to identify authors of listings (e.g. landlord or
tenant) and Hugging Face used to fine-tune a T5-large model to generate
listings given the listing information.

------------------------------------------------------------------------

[London Data - Part 1:
Numbers](https://callumlamont.com/london-data/1_numbers) - The data is
viewed with respect to rental prices, average age of tenant, amenities
of properites etc.

[London Data - Part 2:
Words](https://callumlamont.com/london-data/2_words) - Using
classification (Naive Bayes, SVM, random forests) and NLP techniques
(bag-of-words, word embeddings) to analyse the descriptions advertised
with these property listings.

[London Data - Part
3: Predictions](https://callumlamont.com/london-data/3_generation) - An
attempt to use neural networks to write a description of a flat based on
the listing categories (e.g. number of bedrooms, location, rent).
