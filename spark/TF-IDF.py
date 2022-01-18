from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents (one per line).
rawData = sc.textFile("subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names for later:
documentNames = fields.map(lambda x: x[1])

# Now hash the words in each document to their term frequencies:
hashingTF = HashingTF(100000)  #100K hash buckets just to save some memory
tf = hashingTF.transform(documents)

# At this point we have an RDD of sparse vectors representing each document,
# where each value maps to the term frequency of each unique hash value.

# Let's compute the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# Now we have an RDD of sparse vectors, where each value is the TFxIDF
# of each unique hash value for each document.

# I happen to know that the article for "Abraham Lincoln" is in our data
# set, so let's search for "Gettysburg" (Lincoln gave a famous speech there):

# First, let's figure out what hash value "Gettysburg" maps to by finding the
# index a sparse vector from HashingTF gives us back:
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# Now we will extract the TF*IDF score for Gettsyburg's hash value into
# a new RDD for each document:
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# We'll zip in the document names so we can see which is which:
zippedResults = gettysburgRelevance.zip(documentNames)

# And, print the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())

print("All results----")
for x in zippedResults.takeOrdered(10, key = lambda x: -x[0]):
    print(x)



# OUTPUT of spark-submit

## (base) ➜  spark git:(main) ✗ spark-submit TF-IDF.py
## WARNING: An illegal reflective access operation has occurred
## WARNING: Illegal reflective access by org.apache.spark.unsafe.Platform (file:/opt/homebrew/Cellar/apache-spark/3.2.0/libexec/jars/spark-unsafe_2.12-3.2.0.jar) to constructor java.nio.DirectByteBuffer(long,int)
## WARNING: Please consider reporting this to the maintainers of org.apache.spark.unsafe.Platform
## WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
## WARNING: All illegal access operations will be denied in a future release
## Best document for Gettysburg is:
## (33.13476250917198, 'Abraham Lincoln')
## All results----
## (33.13476250917198, 'Abraham Lincoln')
## (27.612302090976648, 'Abner Doubleday')
## (16.56738125458599, 'American Civil War')
## (0.0, 'Anarchism')
## (0.0, 'Autism')
## (0.0, 'Albedo')
## (0.0, 'A')
## (0.0, 'Alabama')
## (0.0, 'Achilles')
## (0.0, 'Aristotle')