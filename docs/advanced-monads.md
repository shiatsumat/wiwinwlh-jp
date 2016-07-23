# 高度なモナド

## 関数モナド

Haskell を十分に長い間書いている人は、``((->) r)`` のモナドインスタンスという興味深い怪物に出会うことになるかもしれません。これは通常は使うと直感に反するものになりますが、ラッパーを取り除いた読み取りモナドとして考えればかなり単純です。

```haskell
instance Functor ((->) r) where
  fmap = (.)

instance Monad ((->) r) where
  return = const
  f >>= k = \r -> k (f r) r
```

これは、矢印型演算子の前置記法を使っているだけです。

```haskell
import Control.Monad

id' :: (->) a a
id' = id

const' :: (->) a ((->) b a)
const' = const

-- Monad m => a -> m a
fret :: a -> b -> a
fret = return

-- Monad m => m a -> (a -> m b) -> m b
fbind :: (r -> a) -> (a -> (r -> b)) -> (r -> b)
fbind f k = f >>= k

-- Monad m => m (m a) -> m a
fjoin :: (r -> (r -> a)) -> (r -> a)
fjoin = join

fid :: a -> a
fid = const >>= id

-- Functor f => (a -> b) -> f a -> f b
fcompose :: (a -> b) -> (r -> a) -> (r -> b)
fcompose = (.)
```

```haskell
type Reader r = (->) r -- 擬コード

instance Monad (Reader r) where
  return a = \_ -> a
  f >>= k = \ r -> k (f r) r

ask' :: r -> r
ask' = id

asks' :: (r -> a) -> (r -> a)
asks' f = id . f

runReader' :: (r -> a) -> r -> a
runReader' = id
```

## RWSモナド

RWSモナドは[以前](monads.md)議論した3つのモナド――**R**eader［読み取り］、**W**riter［書き留め］、**S**tate［状態］――の機能を組み合わせたものです。``RWST``変換子もあります。

```haskell
runReader :: Reader r a -> r -> a
runWriter :: Writer w a -> (a, w)
runState  :: State s a -> s -> (a, s)
```

これら3つの評価関数は、以下の関数へと合成できます。

```haskell
runRWS  :: RWS r w s a -> r -> s -> (a, s, w)
execRWS :: RWS r w s a -> r -> s -> (s, w)
evalRWS :: RWS r w s a -> r -> s -> (a, w)
```

```haskell
import Control.Monad.RWS

type R = Int
type W = [Int]
type S = Int

computation :: RWS R W S ()
computation = do
  e <- ask
  a <- get
  let b = a + e
  put b
  tell [b]

example = runRWS computation 2 3
```

Writerモナドには遅延性があるというお馴染の但し書きは、RWSにも当てはまります。

## Contモナド

```haskell
runCont :: Cont r a -> (a -> r) -> r
callCC :: MonadCont m => ((a -> m b) -> m a) -> m a
cont :: ((a -> r) -> r) -> Cont r a
```

継続渡しスタイルでは、合成した計算は入れ子になった計算の列から成っています。これらの計算を終了させるのは、継続の連鎖に関数を渡すことで計算全体の結果を返す最終継続です。

```haskell
add :: Int -> Int -> Int
add x y = x + y

add :: Int -> Int -> (Int -> r) -> r
add x y k = k (x + y)
```

```haskell
import Control.Monad
import Control.Monad.Cont

add :: Int -> Int -> Cont k Int
add x y = return $ x + y

mult :: Int -> Int -> Cont k Int
mult x y = return $ x * y

contt :: ContT () IO ()
contt = do
    k <- do
      callCC $ \exit -> do
        lift $ putStrLn "Entry"
        exit $ \_ -> do
          putStrLn "Exit"
    lift $ putStrLn "Inside"
    lift $ k ()

callcc :: Cont String Integer
callcc = do
  a <- return 1
  b <- callCC (\k -> k 2)
  return $ a+b

ex1 :: IO ()
ex1 = print $ runCont (f >>= g) id
  where
    f = add 1 2
    g = mult 3
-- 9

ex2 :: IO ()
ex2 = print $ runCont callcc show
-- "3"

ex3 :: IO ()
ex3 = runContT contt print
-- Entry
-- Inside
-- Exit

main :: IO ()
main = do
  ex1
  ex2
  ex3
```

```haskell
newtype Cont r a = Cont { runCont :: ((a -> r) -> r) }

instance Monad (Cont r) where
  return a       = Cont $ \k -> k a
  (Cont c) >>= f = Cont $ \k -> c (\a -> runCont (f a) k)

class (Monad m) => MonadCont m where
  callCC :: ((a -> m b) -> m a) -> m a

instance MonadCont (Cont r) where
  callCC f = Cont $ \k -> runCont (f (\a -> Cont $ \_ -> k a)) k
```

参照：

* [Wikibooks: Continuation Passing Style](http://en.wikibooks.org/wiki/Haskell/Continuation_passing_style)
* [MonadCont Under the Hood](https://wiki.haskell.org/MonadCont_under_the_hood)

## MonadPlus

選択と失敗を表します。

```haskell
class Monad m => MonadPlus m where
   mzero :: m a
   mplus :: m a -> m a -> m a

instance MonadPlus [] where
   mzero = []
   mplus = (++)

instance MonadPlus Maybe where
   mzero = Nothing

   Nothing `mplus` ys  = ys
   xs      `mplus` _ys = xs
```

MonadPlusは以下の法則によりモノイドを成しています。

```haskell
mzero `mplus` a = a
a `mplus` mzero = a
(a `mplus` b) `mplus` c = a `mplus` (b `mplus` c)
```

```haskell
when :: (Monad m) => Bool -> m () -> m ()
when p s =  if p then s else return ()

guard :: MonadPlus m => Bool -> m ()
guard True  = return ()
guard False = mzero

msum :: MonadPlus m => [m a] -> m a
msum =  foldr mplus mzero
```

```haskell
import Safe
import Control.Monad

list1 :: [(Int,Int)]
list1 = [(a,b) | a <- [1..25], b <- [1..25], a < b]

list2 :: [(Int,Int)]
list2 = do
  a <- [1..25]
  b <- [1..25]
  guard (a < b)
  return $ (a,b)

maybe1 :: String -> String -> Maybe Double
maybe1 a b = do
  a' <- readMay a
  b' <- readMay b
  guard (b' /= 0.0)
  return $ a'/b'

maybe2 :: Maybe Int
maybe2 = msum [Nothing, Nothing, Just 3, Just 4]
```

```haskell
import Control.Monad

range :: MonadPlus m => [a] -> m a
range [] = mzero
range (x:xs) = range xs `mplus` return x

pyth :: Integer -> [(Integer,Integer,Integer)]
pyth n = do
  x <- range [1..n]
  y <- range [1..n]
  z <- range [1..n]
  if x*x + y*y == z*z then return (x,y,z) else mzero

main :: IO ()
main = print $ pyth 15
{-
[ ( 12 , 9 , 15 )
, ( 12 , 5 , 13 )
, ( 9 , 12 , 15 )
, ( 8 , 6 , 10 )
, ( 6 , 8 , 10 )
, ( 5 , 12 , 13 )
, ( 4 , 3 , 5 )
, ( 3 , 4 , 5 )
]
-}
```

## MonadFix

モナディックな計算の不動点です。``mfix f`` は ``f`` のアクションを一度だけ実行し、入力としてフィードバックされた最終的な出力を返します。

```haskell
fix :: (a -> a) -> a
fix f = let x = f x in x

mfix :: (a -> m a) -> m a
```

```haskell
class Monad m => MonadFix m where
   mfix :: (a -> m a) -> m a

instance MonadFix Maybe where
   mfix f = let a = f (unJust a) in a
            where unJust (Just x) = x
                  unJust Nothing  = error "mfix Maybe: Nothing"
```

``-XRecursiveDo``を使えば、通常の do 記法を拡張し、モナド上の再帰的な束縛を許すこともできます。

```haskell
{-# LANGUAGE RecursiveDo #-}

import Control.Applicative
import Control.Monad.Fix

stream1 :: Maybe [Int]
stream1 = do
  rec xs <- Just (1:xs)
  return (map negate xs)

stream2 :: Maybe [Int]
stream2 = mfix $ \xs -> do
  xs' <- Just (1:xs)
  return (map negate xs')
```

## STモナド

STモナドは状態のある計算のスレッドを実現しています。可変な参照を操作することができますが、評価された時純粋な値のみ返すように制限されていて、``s`` のスレッドの ST モナドのみに静的に限定されています。

```haskell
runST :: (forall s. ST s a) -> a
newSTRef :: a -> ST s (STRef s a)
readSTRef :: STRef s a -> ST s a
writeSTRef :: STRef s a -> a -> ST s ()
```

```haskell
import Data.STRef
import Control.Monad
import Control.Monad.ST
import Control.Monad.State.Strict

example1 :: Int
example1 = runST $ do
  x <- newSTRef 0

  forM_ [1..1000] $ \j -> do
    writeSTRef x j

  readSTRef x

example2 :: Int
example2 = runST $ do
  count <- newSTRef 0
  replicateM_ (10^6) $ modifySTRef' count (+1)
  readSTRef count

example3 :: Int
example3 = flip evalState 0 $ do
  replicateM_ (10^6) $ modify' (+1)
  get

modify' :: MonadState a m => (a -> a) -> m ()
modify' f = get >>= (\x -> put $! f x)
```

STモナドを使えば、可変な参照を参照透明な方法で使用する、効率的な純粋関数的データ構造をいくつも生成することができます。

## Freeモナド

```haskell
Pure :: a -> Free f a
Free :: f (Free f a) -> Free f a

liftF :: (Functor f, MonadFree f m) => f a -> m a
retract :: Monad f => Free f a -> f a
```

自由モナドは計算を合成する ``join``［結合］の操作を持つ代わりに、関手の適用から計算を合成します。

```haskell
join :: Monad m => m (m a) -> m a
wrap :: MonadFree f m => f (m a) -> m a
```

最良の例の一つは、発散しうる計算を表現する Partiality［部分性］モナドです。Haskell は非有界な再帰を許しますが、例えば ``Maybe`` 関手から自由モナドを作れば、それを使って[アッカーマン関数](http://ja.wikipedia.org/wiki/%E3%82%A2%E3%83%83%E3%82%AB%E3%83%BC%E3%83%9E%E3%83%B3%E9%96%A2%E6%95%B0)などにおいて呼び出しの深さを固定することができます。

```haskell
import Control.Monad.Fix
import Control.Monad.Free

type Partiality a = Free Maybe a

-- 停止しない
never :: Partiality a
never = fix (Free . Just)

fromMaybe :: Maybe a -> Partiality a
fromMaybe (Just x) = Pure x
fromMaybe Nothing = Free Nothing

runPartiality :: Int -> Partiality a -> Maybe a
runPartiality 0 _ = Nothing
runPartiality _ (Pure a) = Just a
runPartiality _ (Free Nothing) = Nothing
runPartiality n (Free (Just a)) = runPartiality (n-1) a

ack :: Int -> Int -> Partiality Int
ack 0 n = Pure $ n + 1
ack m 0 = Free $ Just $ ack (m-1) 1
ack m n = Free $ Just $ ack m (n-1) >>= ack (m-1)

main :: IO ()
main = do
  let diverge = never :: Partiality ()
  print $ runPartiality 1000 diverge
  print $ runPartiality 1000 (ack 3 4)
  print $ runPartiality 5500 (ack 3 4)
```

自由モナドの他の一般的な使用法は、計算を表現する埋め込みのドメイン固有言語を組み立てることです。IOFree モナドの内部で計算の純粋な描写を組み立てて、自由モナドを使って作用のある IO の計算の翻訳を記述することで、IO モナドのサブセットを実現できます。

```haskell
{-# LANGUAGE DeriveFunctor #-}

import System.Exit
import Control.Monad.Free

data Interaction x
  = Puts String x
  | Gets (Char -> x)
  | Exit
  deriving Functor

type IOFree a = Free Interaction a

puts :: String -> IOFree ()
puts s = liftF $ Puts s ()

get :: IOFree Char
get = liftF $ Gets id

exit :: IOFree r
exit = liftF Exit

gets :: IOFree String
gets = do
  c <- get
  if c == '\n'
    then return ""
    else gets >>= \line -> return (c : line)

-- この IOFree の DSL を潰して、IO モナドのアクションにする
interp :: IOFree a -> IO a
interp (Pure r) = return r
interp (Free x) = case x of
  Puts s t -> putStrLn s >> interp t
  Gets f   -> getChar >>= interp . f
  Exit     -> exitSuccess

echo :: IOFree ()
echo = do
  puts "Enter your name:"
  str <- gets
  puts str
  if length str > 10
    then puts "名前が長いですね"
    else puts "名前が短いですね"
  exit

main :: IO ()
main = interp echo
```

[free](http://hackage.haskell.org/package/free) にあるような実装は、以下のような見た目でしょう。

```haskell
{-# LANGUAGE MultiParamTypeClasses #-}

import Control.Applicative

data Free f a
  = Pure a
  | Free (f (Free f a))

instance Functor f => Monad (Free f) where
  return a     = Pure a
  Pure a >>= f = f a
  Free f >>= g = Free (fmap (>>= g) f)

class Monad m => MonadFree f m  where
  wrap :: f (m a) -> m a

liftF :: (Functor f, MonadFree f m) => f a -> m a
liftF = wrap . fmap return

iter :: Functor f => (f a -> a) -> Free f a -> a
iter _ (Pure a) = a
iter phi (Free m) = phi (iter phi <$> m)

retract :: Monad f => Free f a -> f a
retract (Pure a) = return a
retract (Free as) = as >>= retract
```

参照：

* [Monads for Free!](http://www.andres-loeh.de/Free.pdf)
* [I/O is not a Monad](http://r6.ca/blog/20110520T220201Z.html)

## 指標付きモナド

指標 (index) 付きモナドは、モナドを一般化して、クラスに余分な型引数を追加したものです。この引数は、モナディックな実装の計算や構造についての情報を保持しています。

```haskell
class IxMonad md where
  return :: a -> md i i a
  (>>=) :: md i m a -> (a -> md m o b) -> md i o b
```

標準的なユースケースは、ありきたりな State を少し変えたもので、モナドの内部で途中の段階で状態の型を変えることが出来るものです。これは実は、リソース管理に絡んだいくつかの問題を扱うのに、とても便利なのです。余分な指標の引数により、コンパイル時に指標の引数における特定の状態遷移を許可したり制限したりすることで、モナディックなアクションの列を静的に実行する余地が生じます。

これをより使いやすくするには、やや難解である ``-XRebindableSyntax`` を使います。これを使えば、do 記法や if-then-else 構文をオーバーロードし、モジュール内限定で代替の定義を与えることができます。

```haskell
{-# LANGUAGE RebindableSyntax #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

import Data.IORef
import Data.Char
import Prelude hiding (fmap, (>>=), (>>), return)
import Control.Applicative

newtype IState i o a = IState { runIState :: i -> (a, o) }

evalIState :: IState i o a -> i -> a
evalIState st i = fst $ runIState st i

execIState :: IState i o a -> i -> o
execIState st i = snd $ runIState st i

ifThenElse :: Bool -> a -> a -> a
ifThenElse b i j = case b of
  True -> i
  False -> j

return :: a -> IState s s a
return a = IState $ \s -> (a, s)

fmap :: (a -> b) -> IState i o a -> IState i o b
fmap f v = IState $ \i -> let (a, o) = runIState v i
                          in (f a, o)

join :: IState i m (IState m o a) -> IState i o a
join v = IState $ \i -> let (w, m) = runIState v i
                        in runIState w m

(>>=) :: IState i m a -> (a -> IState m o b) -> IState i o b
v >>= f = IState $ \i -> let (a, m) = runIState v i
                         in runIState (f a) m

(>>) :: IState i m a -> IState m o b -> IState i o b
v >> w = v >>= \_ -> w

get :: IState s s s
get = IState $ \s -> (s, s)

gets :: (a -> o) -> IState a o a
gets f = IState $ \s -> (s, f s)

put :: o -> IState i o ()
put o = IState $ \_ -> ((), o)

modify :: (i -> o) -> IState i o ()
modify f = IState $ \i -> ((), f i)



data Locked = Locked
data Unlocked = Unlocked

type Stateful a = IState a Unlocked a

acquire :: IState i Locked ()
acquire = put Locked

-- ロックが保持されている場合に限りロックを解放することができ、
-- ロックが保持されていない状態でロックを解放しようとすると
-- 型エラーとなってしまう
release :: IState Locked Unlocked ()
release = put Unlocked

-- 静的に、リソースの不適切な処理を禁じている
lockExample :: Stateful a
lockExample = do ptr <- get  :: IState a a a
                 acquire     :: IState a Locked ()
                 -- ...
                 release     :: IState Locked Unlocked ()
                 return ptr

-- Couldn't match type `Locked' with `Unlocked'
-- In a stmt of a 'do' block: return ptr
failure1 :: Stateful a
failure1 = do ptr <- get
              acquire
              return ptr -- didn't release

-- Couldn't match type `a' with `Locked'
-- In a stmt of a 'do' block: release
failure2 :: Stateful a
failure2 = do ptr <- get
              release -- didn't acquire
              return ptr

-- 終了時にロックが解放されていることを静的に保証しつつ、
-- 結果として得られる状態を評価する
evalReleased :: IState i Unlocked a -> i -> a
evalReleased f st = evalIState f st

example :: IO (IORef Integer)
example = evalReleased <$> pure lockExample <*> newIORef 0
```

参照：

* [Fun with Indexed monads](http://www.cl.cam.ac.uk/~dao29/ixmonad/ixmonad-fita14.pdf)
