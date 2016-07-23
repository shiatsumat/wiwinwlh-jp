# データフォーマット

## JSON

Aeson is library for efficient parsing and generating JSON.

```haskell
decode :: FromJSON a => ByteString -> Maybe a
encode :: ToJSON a => a -> ByteString
eitherDecode :: FromJSON a => ByteString -> Either String a

fromJSON :: FromJSON a => Value -> Result a
toJSON :: ToJSON a => a -> Value
```

We'll work with this contrived example:

```json
{
    "id": 1,
    "name": "A green door",
    "price": 12.50,
    "tags": ["home", "green"],
    "refs": {
      "a": "red",
      "b": "blue"
    }
}
```

Aeson uses several high performance data structures (Vector, Text, HashMap) by default instead of the naive
versions so typically using Aeson will require that us import them and use ``OverloadedStrings`` when
indexing into objects.

```haskell
type Object = HashMap Text Value

type Array = Vector Value

-- | A JSON value represented as a Haskell value.
data Value = Object !Object
           | Array !Array
           | String !Text
           | Number !Scientific
           | Bool !Bool
           | Null
```

See: [Aeson Documentation](http://hackage.haskell.org/package/aeson)

**Unstructured**

In dynamic scripting languages it's common to parse amorphous blobs of JSON without any a priori structure and
then handle validation problems by throwing exceptions while traversing it. We can do the same using Aeson and
the Maybe monad.

```haskell
{-# LANGUAGE OverloadedStrings #-}

import Data.Text
import Data.Aeson
import Data.Vector
import qualified Data.HashMap.Strict as M
import qualified Data.ByteString.Lazy as BL

-- Pull a key out of an JSON object.
(^?) :: Value -> Text -> Maybe Value
(^?) (Object obj) k = M.lookup k obj
(^?) _ _ = Nothing

-- Pull the ith value out of a JSON list.
ix :: Value -> Int -> Maybe Value
ix (Array arr) i = arr !? i
ix _ _ = Nothing

readJSON str = do
  obj <- decode str
  price <- obj ^? "price"
  refs  <- obj ^? "refs"
  tags  <- obj ^? "tags"
  aref  <- refs ^? "a"
  tag1  <- tags `ix` 0
  return (price, aref, tag1)

main :: IO ()
main = do
  contents <- BL.readFile "example.json"
  print $ readJSON contents
```

**Structured**

This isn't ideal since we've just smeared all the validation logic across our traversal logic instead of
separating concerns and handling validation in separate logic. We'd like to describe the structure before-hand
and the invalid case separately. Using Generic also allows Haskell to automatically write the serializer and
deserializer between our datatype and the JSON string based on the names of record field names.

```haskell
{-# LANGUAGE DeriveGeneric #-}

import Data.Text
import Data.Aeson
import GHC.Generics
import qualified Data.ByteString.Lazy as BL

import Control.Applicative

data Refs = Refs
  { a :: String
  , b :: String
  } deriving (Show,Generic)

data Data = Data
  { id    :: Int
  , name  :: Text
  , price :: Int
  , tags  :: [String]
  , refs  :: Refs
  } deriving (Show,Generic)

instance FromJSON Data
instance FromJSON Refs
instance ToJSON Data
instance ToJSON Refs

main :: IO ()
main = do
  contents <- BL.readFile "example.json"
  let Just dat = decode contents
  print $ name dat
  print $ a (refs dat)
```

Now we get our validated JSON wrapped up into a nicely typed Haskell ADT.

```haskell
Data
  { id = 1
  , name = "A green door"
  , price = 12
  , tags = [ "home" , "green" ]
  , refs = Refs { a = "red" , b = "blue" }
  }
```

The functions ``fromJSON`` and ``toJSON`` can be used to convert between this sum type and regular Haskell
types with.

```haskell
data Result a = Error String | Success a
```

```haskell
λ: fromJSON (Bool True) :: Result Bool
Success True

λ: fromJSON (Bool True) :: Result Double
Error "when expecting a Double, encountered Boolean instead"
```

## CSV

Cassava is an efficient CSV parser library. We'll work with this tiny snippet from the iris dataset:

```perl
sepal_length,sepal_width,petal_length,petal_width,plant_class
5.1,3.5,1.4,0.2,Iris-setosa
5.0,2.0,3.5,1.0,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
```

**Unstructured**

Just like with Aeson if we really want to work with unstructured data the library accommodates this.

```haskell
import Data.Csv

import Text.Show.Pretty

import qualified Data.Vector as V
import qualified Data.ByteString.Lazy as BL

type ErrorMsg = String
type CsvData = V.Vector (V.Vector BL.ByteString)

example :: FilePath -> IO (Either ErrorMsg CsvData)
example fname = do
  contents <- BL.readFile fname
  return $ decode NoHeader contents
```

We see we get the nested set of stringy vectors:

```haskell
[ [ "sepal_length"
  , "sepal_width"
  , "petal_length"
  , "petal_width"
  , "plant_class"
  ]
, [ "5.1" , "3.5" , "1.4" , "0.2" , "Iris-setosa" ]
, [ "5.0" , "2.0" , "3.5" , "1.0" , "Iris-versicolor" ]
, [ "6.3" , "3.3" , "6.0" , "2.5" , "Iris-virginica" ]
]
```

**Structured**

Just like with Aeson we can use Generic to automatically write the deserializer between our CSV data and our
custom datatype.

```haskell
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

import Data.Csv
import GHC.Generics
import qualified Data.Vector as V
import qualified Data.ByteString.Lazy as BL

data Plant = Plant
  { sepal_length :: Double
  , sepal_width  :: Double
  , petal_length :: Double
  , petal_width  :: Double
  , plant_class :: String
  } deriving (Generic, Show)

instance FromNamedRecord Plant
instance ToNamedRecord Plant

type ErrorMsg = String
type CsvData = (Header, V.Vector Plant)

parseCSV :: FilePath -> IO (Either ErrorMsg CsvData)
parseCSV fname = do
  contents <- BL.readFile fname
  return $ decodeByName contents

main = parseCSV "iris.csv" >>= print
```

And again we get a nice typed ADT as a result.

```haskell
[ Plant
    { sepal_length = 5.1
    , sepal_width = 3.5
    , petal_length = 1.4
    , petal_width = 0.2
    , plant_class = "Iris-setosa"
    }
, Plant
    { sepal_length = 5.0
    , sepal_width = 2.0
    , petal_length = 3.5
    , petal_width = 1.0
    , plant_class = "Iris-versicolor"
    }
, Plant
    { sepal_length = 6.3
    , sepal_width = 3.3
    , petal_length = 6.0
    , petal_width = 2.5
    , plant_class = "Iris-virginica"
    }
]
```
