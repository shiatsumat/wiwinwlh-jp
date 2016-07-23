# TextとByteString

## はじめに

Haskell のデフォルトの文字列型は、割と愚直な、文字の連結リストであり、小さい識別子に使う場合は全く問題ないですが、大量のデータを処理するのには向いていません。

```haskell
type String = [Char]
```

よりパフォーマンスに敏感な場合、テキストデータを処理するライブラリが2つあります。``text`` と ``bytestring`` です。``-XOverloadedStrings`` 拡張を付ければ、文字列リテラルを明示的なパッキングを行うことなくオーバーロードし、文字列リテラルとして Haskell のソースに書き、型クラス ``IsString`` を通してオーバーロードすることができます。

```haskell
class IsString a where
  fromString :: String -> a
```

例えば：

```haskell
λ: :type "foo"
"foo" :: [Char]

λ: :set -XOverloadedStrings

λ: :type "foo"
"foo" :: IsString a => a
```

## Text

Text型はユニコード文字をいくつかパックした何かです。

```haskell
pack :: String -> Text
unpack :: Text -> String
```

```haskell
{-# LANGUAGE OverloadedStrings #-}

import qualified Data.Text as T

-- pack から
myTStr1 :: T.Text
myTStr1 = T.pack ("foo" :: String)

-- オーバーロードした文字列リテラルから
myTStr2 :: T.Text
myTStr2 = "bar"
```

参照：

* [Text](http://hackage.haskell.org/package/text-1.2.0.4/docs/Data-Text.html)

## Textのビルダ

```haskell
toLazyText :: Builder -> Data.Text.Lazy.Internal.Text
fromLazyText :: Data.Text.Lazy.Internal.Text -> Builder
```

Text のビルダを使えば、怠惰な Text 型を、文字列やリストのような非効率なものを中間物として経由することなく、効率的なモノイドの構成を作ることができます。

```haskell
{-# LANGUAGE OverloadedStrings #-}

import Data.Monoid (mconcat, (<>))

import Data.Text.Lazy.Builder (Builder, toLazyText)
import Data.Text.Lazy.Builder.Int (decimal)
import qualified Data.Text.Lazy.IO as L

beer :: Int -> Builder
beer n = decimal n <> " bottles of beer on the wall.\n"

wall :: Builder
wall = mconcat $ fmap beer [1..1000]

main :: IO ()
main = L.putStrLn $ toLazyText wall
```

## ByteString

ByteString は正格評価か遅延評価を用いる非ボックス文字の配列です。

```haskell
pack :: String -> ByteString
unpack :: ByteString -> String
```

```haskell
{-# LANGUAGE OverloadedStrings #-}

import qualified Data.ByteString as S
import qualified Data.ByteString.Char8 as S8

-- From pack
bstr1 :: S.ByteString
bstr1 = S.pack ("foo" :: String)

-- From overloaded string literal.
bstr2 :: S.ByteString
bstr2 = "bar"
```

参照：

* [Bytestring: Bits and Pieces](https://www.fpcomplete.com/school/to-infinity-and-beyond/pick-of-the-week/bytestring-bits-and-pieces)
* [ByteString](http://hackage.haskell.org/package/bytestring-0.10.4.0/docs/Data-ByteString.html)

## printf

Haskell には、C のスタイルの可変個引数の ``printf`` 関数もあります。

```haskell
import Data.Text
import Text.Printf

a :: Int
a = 3

b :: Double
b = 3.14159

c :: String
c = "haskell"

example :: String
example = printf "(%i, %f, %s)" a b c
-- "(3, 3.14159, haskell)"
```

## 多重定義されたリスト

データ構造のライブラリがリストから様々な構造を構成するために ``toList`` や ``fromList`` を公開しているのはよく見かけます。GHC 7.8 の時点では、型クラス ``IsList`` を使って表層の言語でリストの構文をオーバーロードすることができます。

```haskell
class IsList l where
  type Item l
  fromList  :: [Item l] -> l
  toList    :: l -> [Item l]

instance IsList [a] where
  type Item [a] = a
  fromList = id
  toList   = id
```

```haskell
λ: :type [1,2,3]
[1,2,3] :: (Num (Item l), IsList l) => l
```

```haskell
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeFamilies #-}

import qualified Data.Map as Map
import GHC.Exts (IsList(..))

instance (Ord k) => IsList (Map.Map k v) where
  type Item (Map.Map k v) = (k,v)
  fromList = Map.fromList
  toList = Map.toList

example1 :: Map.Map String Int
example1 = [("a", 1), ("b", 2)]
```
