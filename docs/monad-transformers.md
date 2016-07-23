# モナド変換子

## mtlとtransformers

[前章](monads.md)でのモナドの説明は、罪のない嘘が少々交じっています。現代の Haskell のモナドライブラリは、概して、モナド変換子を使って書いた、より一般性のある形を使うのです。モナド変換子を使えば、モナドを合成して合成モナドを作ることができます。先ほど触れたモナドは、変換子の形のものを Identity モナドと合成した、特定の場合に特化したものとして扱うことができます。

モナド | 変換子 | 型 | 変換子の型
--- | --- | --- | ---
Maybe | MaybeT | ``Maybe a`` | ``m (Maybe a)``
Reader | ReaderT | ``r -> a`` | ``r -> m a``
Writer | WriterT | ``(a,w)`` | ``m (a,w)``
State | StateT | ``s -> (a,s)`` | ``s -> m (a,s)``

```haskell
type State  s = StateT  s Identity
type Writer w = WriterT w Identity
type Reader r = ReaderT r Identity

instance Monad m => MonadState s (StateT s m)
instance Monad m => MonadReader r (ReaderT r m)
instance (Monoid w, Monad m) => MonadWriter w (WriterT w m)
```

一般性の観点からは、mtl ライブラリはこれらのモナドに対する共通のインターフェースとして最も一般性がありますが、このライブラリ自体、上述の”基本的”なモナドを変換子へと一般化している transformers ライブラリに依存しています。

## 変換子

中核では、モナド変換子はモナディックな計算をスタック上で入れ子に出来るようにしています。ここで、様々なレベルの間で値をやりとりするための ``lift``［持ち上げ］というインターフェースが提供されています。

```haskell
lift :: (Monad m, MonadTrans t) => m a -> t m a
liftIO :: MonadIO m => IO a -> m a
```

```haskell
class MonadTrans t where
    lift :: Monad m => m a -> t m a

class (Monad m) => MonadIO m where
    liftIO :: IO a -> m a

instance MonadIO IO where
    liftIO = id
```

基盤にあるモナドクラスが法則を持つように、モナド変換子もいくつかの法則を持っています。

**法則 1**

```haskell
lift . return = return
```

**法則 2**

```haskell
lift (m >>= f) = lift m >>= (lift . f)
```

次のようにも書けます。

**法則 1**

```haskell
  lift (return x)

= return x
```

**法則 2**

```haskell
  do x <- lift m
     lift (f x)

= lift $ do x <- m
            f x
```

変換子の合成は外側から内側へと為されますが、ほどいていく際には内側から外側へと行く、と覚えておくと便利でしょう。

![](https://raw.githubusercontent.com/sdiehl/wiwinwlh/master/img/transformer_unroll.png)

参照：

* [Monad Transformers: Step-By-Step](http://www.cs.virginia.edu/~wh5a/personal/Transformers.pdf)

## ReaderT

例えば、読み取りモナドには 3 つの形が可能性として存在します。1 番目は Haskell 98 のもので、今はもう使われませんが、教育上は有用です。*transformers* の変種と *mtl* の変種を合わせて紹介しています。

**Reader**

```haskell
newtype Reader r a = Reader { runReader :: r -> a }

instance MonadReader r (Reader r) where
  ask       = Reader id
  local f m = Reader $ runReader m . f
```

**ReaderT**

```haskell
newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }

instance (Monad m) => Monad (ReaderT r m) where
  return a = ReaderT $ \_ -> return a
  m >>= k  = ReaderT $ \r -> do
      a <- runReaderT m r
      runReaderT (k a) r

instance MonadTrans (ReaderT r) where
    lift m = ReaderT $ \_ -> m
```

**MonadReader**

```haskell
class (Monad m) => MonadReader r m | m -> r where
  ask   :: m r
  local :: (r -> r) -> m a -> m a

instance (Monad m) => MonadReader r (ReaderT r m) where
  ask       = ReaderT return
  local f m = ReaderT $ \r -> runReaderT m (f r)
```

ですから、``ask`` の 3 つの変種は以下のようになると仮定できます。

```haskell
ask :: Reader r a
ask :: Monad m => ReaderT r m r
ask :: MonadReader r m => m r
```

実用上は、最後のものだけが現代の Haskell では使われています。

## 基本

最も基本的な使い方では、外側の層に、T の付いた変種であるモナド変換子を使って、各層の間で明示的に ``lift`` し値を ``return`` せねばなりません。モナドは ``(* -> *)`` の種を持つので、モナドを受け取ってモナドにするモナド変換子は ``((* -> *) -> * -> *)`` の種を持ちます。

```haskell
Monad (m :: * -> *)
MonadTrans (t :: (* -> *) -> * -> *)
```

例えば、Reader と Maybe の両方のモナドを用いる合成計算を作りたければ、``ReaderT`` の内側に ``Maybe`` を入れて ``ReaderT t Maybe a`` を作ればいいのです。

```haskell
import Control.Monad.Reader

type Env = [(String, Int)]
type Eval a = ReaderT Env Maybe a

data Expr
  = Val Int
  | Add Expr Expr
  | Var String
  deriving (Show)

eval :: Expr -> Eval Int
eval ex = case ex of

  Val n -> return n

  Add x y -> do
    a <- eval x
    b <- eval y
    return (a+b)

  Var x -> do
    env <- ask
    val <- lift (lookup x env)
    return val

env :: Env
env = [("x", 2), ("y", 5)]

ex1 :: Eval Int
ex1 = eval (Add (Val 2) (Add (Val 1) (Var "x")))

example1, example2 :: Maybe Int
example1 = runReaderT ex1 env
example2 = runReaderT ex1 []
```

この方法の根本的な限界は、``lift.lift.lift`` や ``return.return.return`` をたくさんせねばならないということです。

## newtype導出

newtype を使えば、単一コンストラクタのデータ型を、新たな別の型として、（単一コンストラクタの代数的データ型と違い）ボックスの出し入れによる実行時のオーバーヘッド無しに使うことができます。文字列や数値に対する newtype のラッパーを使えば、しばしば事故的なエラーを劇的に減らすことができます。``-XGeneralizedNewtypeDeriving`` を使えば、基盤にある型のインスタンスの機能を復元することもできます。

```haskell
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

newtype Velocity = Velocity { unVelocity :: Double }
  deriving (Eq, Ord)

v :: Velocity
v = Velocity 2.718

x :: Double
x = 6.636

-- 実行時には同じ種類の値ですが、コンパイル時には型エラーが捕捉されます！
err = v + x

newtype Quantity v a = Quantity a
  deriving (Eq, Ord, Num, Show)

data Haskeller
type Haskellers = Quantity Haskeller Int

a = Quantity 2 :: Haskellers
b = Quantity 6 :: Haskellers

totalHaskellers :: Haskellers
totalHaskellers = a + b
```

```haskell
Couldn't match type `Double' with `Velocity'
Expected type: Velocity
  Actual type: Double
In the second argument of `(+)', namely `x'
In the expression: v + x
```

newtype導出をmtlライブラリの型クラスと一緒に使えば、変換のスタックで明示的に持ち上げる必要のない、平坦な変換子の型を作れます。例えば、下にあるのは読み取り、書き留め、状態のモナドからなる小さなスタックの機械です。

```haskell
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

import Control.Monad.Reader
import Control.Monad.Writer
import Control.Monad.State

type Stack   = [Int]
type Output  = [Int]
type Program = [Instr]

type VM a = ReaderT Program (WriterT Output (State Stack)) a

newtype Comp a = Comp { unComp :: VM a }
  deriving (Monad, MonadReader Program, MonadWriter Output, MonadState Stack)

data Instr = Push Int | Pop | Puts

evalInstr :: Instr -> Comp ()
evalInstr instr = case instr of
  Pop    -> modify tail
  Push n -> modify (n:)
  Puts   -> do
    tos <- gets head
    tell [tos]

eval :: Comp ()
eval = do
  instr <- ask
  case instr of
    []     -> return ()
    (i:is) -> evalInstr i >> local (const is) eval

execVM :: Program -> Output
execVM = flip evalState [] . execWriterT . runReaderT (unComp eval)

program :: Program
program = [
     Push 42,
     Push 27,
     Puts,
     Pop,
     Puts,
     Pop
  ]

main :: IO ()
main = mapM_ print $ execVM program
```

newtype コンストラクタでのパターンマッチングはコンパイルされると無くなります。例えば、``extractB`` 関数は、``extractA`` と違い、``MkB`` コンストラクタを調べません。``MkB`` は実行時には存在せず、コンパイル時にあるものにすぎないからです。

```haskell
data A = MkA Int
newtype B = MkB Int

extractA :: A -> Int
extractA (MkA x) = x

extractB :: B -> Int
extractB (MkB x) = x
```

## 効率

2 番目のモナド変換子の法則は、持ち上げ操作を連続して行ったものを配列 (sequence) することは、外側のモナドへと結果を持ち上げることと、意味論的に等価である、ということを保証しています。

```haskell
do x <- lift m  ==  lift $ do x <- m
   lift (f x)                 f x
```

同じ結果を出すことは保証されていますが、モナドのレベルの間で結果を持ち上げる操作はコスト無しに出来ることでは無く、モナド上で走査やループする関数を行うと、しばしばコストがかさむのです。例えば、下記の 3 つの左辺の関数は全て右辺の関数より非効率です。右辺の関数は基盤のモナドで束縛をしていますが、左辺の関数は毎回毎回持ち上げをしているのです。

```haskell
-- Less Efficient      More Efficient
forever (lift m)    == lift (forever m)
mapM_ (lift . f) xs == lift (mapM_ f xs)
forM_ xs (lift . f) == lift (forM_ xs f)
```

## モナド射

```haskell
lift :: Monad m => m a -> t m a
```

```haskell
hoist :: Monad m => (forall a. m a -> n a) -> t m b -> t n b
embed :: Monad n => (forall a. m a -> t n a) -> t m b -> t n b
squash :: (Monad m, MMonad t) => t (t m) a -> t m a
```

未完成

参照：

* [mmorph](https://hackage.haskell.org/package/mmorph)
