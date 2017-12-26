import tensorflow as tf


"""
Raw features for the census dataset. 
The categorical columns should probably be transformed to a dense form
by applying indicator_column() or embedding_column() from tf.feature_column.
"""


num = tf.feature_column.numeric_column
cat = tf.feature_column.categorical_column_with_vocabulary_list
cat_with_hash = tf.feature_column.categorical_column_with_hash_bucket


RAW_FEATURE_COLUMNS = [
    # Continuous columns.
    num("age"),
    num("education_num"),
    num("capital_gain"),
    num("capital_loss"),
    num("hours_per_week"),

    # Categorical columns.
    cat("gender", [" Female", " Male"]),
    cat("race",
        [" Amer-Indian-Eskimo", " Asian-Pac-Islander",
         " Black", " Other", " White"]),
    cat("education",
        [" Bachelors", " HS-grad", " 11th", " Masters", " 9th",
         " Some-college", " Assoc-acdm", " Assoc-voc", " 7th-8th",
         " Doctorate", " Prof-school", " 5th-6th", " 10th",
         " 1st-4th", " Preschool", " 12th"]),
    cat("marital_status",
        [" Married-civ-spouse", " Divorced", " Married-spouse-absent",
         " Never-married", " Separated", " Married-AF-spouse", " Widowed"]),
    cat("relationship",
        [" Husband", " Not-in-family", " Wife", " Own-child", " Unmarried",
         " Other-relative"]),
    cat("workclass",
        [" Self-emp-not-inc", " Private", " State-gov",
         " Federal-gov", " Local-gov", " ?", " Self-emp-inc",
         " Without-pay", " Never-worked"]),
    cat_with_hash("occupation", hash_bucket_size=100, dtype=tf.string),
    cat_with_hash("native_country", hash_bucket_size=100, dtype=tf.string)
]

LABEL_VALUES = [" <=50K", " >50K"]
