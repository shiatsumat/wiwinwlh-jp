# アプリカティブ

## はじめに

モナドと同様、アプリカティブ［適用可能関手］は抽象的構造ですが、関手とモナドの中間の一般性を持つ幅広い種類の計算を扱います。

```haskell
pure :: Applicative f => a -> f a
(<$>) :: Functor f => (a -> b) -> f a -> f b
(<*>) :: f (a -> b) -> f a -> f b
```

GHC 7.6 の時点では、Applicative はこのように定義されています。

```haskell
class Functor f => Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

(<$>) :: Functor f => (a -> b) -> f a -> f b
(<$>) = fmap
```

法則は以下の通りです。

```haskell
pure id <*> v = v
pure f <*> pure x = pure (f x)
u <*> pure y = pure ($ y) <*> u
u <*> (v <*> w) = pure (.) <*> u <*> v <*> w
```

例として、Maybe のインスタンスを考えましょう。

```haskell
instance Applicative Maybe where
  pure              = Just
  Nothing <*> _     = Nothing
  _ <*> Nothing     = Nothing
  Just f <*> Just x = Just (f x)
```

大まかに言って、``m >>= return . f`` を使いたいとき、私たちが欲しいのはおそらくアプリカティブな関手であり、モナドではないのです。

```haskell
import Network.HTTP
import Control.Applicative ((<$>),(<*>))

example1 :: Maybe Integer
example1 = (+) <$> m1 <*> m2
  where
    m1 = Just 3
    m2 = Nothing
-- Nothing

example2 :: [(Int, Int, Int)]
example2 = (,,) <$> m1 <*> m2 <*> m3
  where
    m1 = [1,2]
    m2 = [10,20]
    m3 = [100,200]
-- [(1,10,100),(1,10,200),(1,20,100),(1,20,200),(2,10,100),(2,10,200),(2,20,100),(2,20,200)]

example3 :: IO String
example3 = (++) <$> fetch1 <*> fetch2
  where
    fetch1 = simpleHTTP (getRequest "http://www.fpcomplete.com/") >>= getResponseBody
    fetch2 = simpleHTTP (getRequest "http://www.haskell.org/") >>= getResponseBody
```

``f <$> a <*> b ...`` というパターンは頻繁に現れるので、アプリカティブを一定個数の引数のアプリカティブを持ち上げる関数のグループがあります。このパターンはモナドでも頻繁に表れます（``liftM``, ``liftM2``, ``liftM3``）。

```haskell
liftA :: Applicative f => (a -> b) -> f a -> f b
liftA f a = pure f <*> a

liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
liftA2 f a b = f <$> a <*> b

liftA3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
liftA3 f a b c = f <$> a <*> b <*> c
```

アプリカティブには ``*>`` と ``<*`` という関数もあります。これらは、引数の片方の値を無視してアプリカティブのアクションを順に実行するものです。``*>`` は左の値を無視し、``<*`` は右の値を無視します。例えば、モナディックなパーサコンビネータのライブラリでは、``*>`` は最初の引数のパーサでパースして、二番目の引数の値を返します。

アプリカティブの関数 ``<$>`` と ``<*>`` は、モナドに対する ``liftM`` や ``ap`` を一般化したものです。

```haskell
import Control.Monad
import Control.Applicative

data C a b = C a b

mnd :: Monad m => m a -> m b -> m (C a b)
mnd a b = C `liftM` a `ap` b

apl :: Applicative f => f a -> f b -> f (C a b)
apl a b = C <$> a <*> b
```

参照：

* [Applicative Programming with Effects](http://www.soi.city.ac.uk/~ross/papers/Applicative.pdf)

## 型クラスの階層

原則として、全てのモナドはアプリカティブな関手から（その帰結として関手から）生まれますが、歴史的な理由からアプリカティブはモナド型クラスのスーパークラスではありません。仮定上の話ですが、Prelude を修正するとこのようになるかもしれません。

```haskell
class Functor f where
  fmap :: (a -> b) -> f a -> f b

class Functor f => Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

class Applicative m => Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
  ma >>= f = join (fmap f ma)

return :: Applicative m => a -> m a
return = pure

join :: Monad m => m (m a) -> m a
join x = x >>= id
```

参照：

[Functor-Applicative-Monad Proposal](http://www.haskell.org/haskellwiki/Functor-Applicative-Monad_Proposal)

## オルタナティブ

オルタナティブはアプリカティブクラスに、零元と、零元を尊重する［零元が単位元である］結合的な二項演算を加えて拡張したものです。

```haskell
class Applicative f => Alternative f where
  -- | The identity of '<|>'
  empty :: f a
  -- | An associative binary operation
  (<|>) :: f a -> f a -> f a
  -- | One or more.
  some :: f a -> f [a]
  -- | Zero or more.
  many :: f a -> f [a]

optional :: Alternative f => f a -> f (Maybe a)
```

```haskell
instance Alternative Maybe where
    empty = Nothing
    Nothing <|> r = r
    l       <|> _ = l

instance Alternative [] where
    empty = []
    (<|>) = (++)
```

```haskell
λ: foldl1 (<|>) [Nothing, Just 5, Just 3]
Just 5
```

これらのインスタンスは、オルタナティブの演算子により代替のパースの選択肢をモデル化できる場合に、パーサーで非常に頻繁に現れます。

## 可変個引数関数

型クラスの驚くべき応用の一つは、関数型に対してインスタンスを定義することで任意個の引数を受け取る関数を作れるようになることです。引数はどんな型でも構いませんが、引数を集めた結果は、単一の型に変換するか、アンパックして和型にしなければなりません。

```haskell
{-# LANGUAGE FlexibleInstances #-}

class Arg a where
  collect' :: [String] -> a

-- extract to IO
instance Arg (IO ()) where
  collect' acc = mapM_ putStrLn acc

-- extract to [String]
instance Arg [String] where
  collect' acc = acc

instance (Show a, Arg r) => Arg (a -> r) where
  collect' acc = \x -> collect' (acc ++ [show x])

collect :: Arg t => t
collect = collect' []

example1 :: [String]
example1 = collect 'a' 2 3.0

example2 :: IO ()
example2 = collect () "foo" [1,2,3]
```

参照：

* [Polyvariadic functions](http://okmij.org/ftp/Haskell/polyvariadic.html)

## 圏

圏は、恒等元（恒等射）と、結合的な合成演算の概念を含む代数的構造です。

```haskell
class Category cat where
  id :: cat a a
  (.) :: cat b c -> cat a b -> cat a c
```

```haskell
instance Category (->) where
  id = Prelude.id
  (.) = (Prelude..)
```

```haskell
(<<<) :: Category cat => cat b c -> cat a b -> cat a c
(<<<) = (.)

(>>>) :: Category cat => cat a b -> cat b c -> cat a c
f >>> g = g . f
```

## アロー

アローは、圏に積の概念を加えて拡張したものです。

```haskell
class Category a => Arrow a where
  arr :: (b -> c) -> a b c
  first :: a b c -> a (b,d) (c,d)
  second :: a b c -> a (d,b) (d,c)
  (***) :: a b c -> a b' c' -> a (b,b') (c,c')
  (&&&) :: a b c -> a b c' -> a b (c,c')
```

標準的な例は、関数に対するものです。

```haskell
instance Arrow (->) where
  arr f = f
  first f = f *** id
  second f = id *** f
  (***) f g ~(x,y) = (f x, g y)
```

この形式では、複数引数の関数はアローのコンビネータを使って、よりポイントフリーな形で操作できます。例えば、ヒストグラムの関数にはいい感じのワンライナーがあります。

```haskell
histogram :: Ord a => [a] -> [(a, Int)]
histogram = map (head &&& length) . group . sort
```

```haskell
λ: histogram "Hello world"
[(' ',1),('H',1),('d',1),('e',1),('l',3),('o',2),('r',1),('w',1)]
```

**アロー記法**

以下のものは等価です。

```haskell
{-# LANGUAGE Arrows #-}

addA :: Arrow a => a b Int -> a b Int -> a b Int
addA f g = proc x -> do
                y <- f -< x
                z <- g -< x
                returnA -< y + z
```

```haskell
addA f g = arr (\ x -> (x, x)) >>>
           first f >>> arr (\ (y, x) -> (x, y)) >>>
           first g >>> arr (\ (z, y) -> y + z)
```

```haskell
addA f g = f &&& g >>> arr (\ (y, z) -> y + z)
```

実用上ではこの記法はそんなに使われていないので、将来、廃止されるかもしれません。

参照：

* [Arrow Notation](https://downloads.haskell.org/~ghc/7.8.3/docs/html/users_guide/arrow-notation.html)

## 双関手

双関手 (bifunctor) は関手を一般化して、2 つの引数を付けた型を含むようにしたものであり、各引数に対して 2 つの map 関数を持ちます。

```haskell
class Bifunctor p where
  bimap :: (a -> b) -> (c -> d) -> p a c -> p b d
  first :: (a -> b) -> p a c -> p b c
  second :: (b -> c) -> p a b -> p a c
```

双関手則は、通常の関手の自然な一般化です。即ち、通常の方法で恒等関数と合成関数を尊重しているのです。

```haskell
bimap id id ≡ id
first id ≡ id
second id ≡ id
```

```haskell
bimap f g ≡ first f . second g
```

標準的な例はペア（2 つ組）に対するものです。

```haskell
λ: first (+1) (1,2)
(2,2)
λ: second (+1) (1,2)
(1,3)
λ: bimap (+1) (+1) (1,2)
(2,3)

λ: first (+1) (Left 3)
Left 4
λ: second (+1) (Left 3)
Left 3
λ: second (+1) (Right 3)
Right 4
```
