# エラー処理

## Control.Exception

エラーを扱う低レベルの（とても危険な）方法は、``throw``［例外の発生］関数と ``catch``［例外の捕捉］関数を使うことです。これらの関数を使えば、純粋なコードで拡張可能な例外を投げることができますが、得られる例外は IO 内で捕捉します。特に指摘しておくべきなのは、``throw`` の返す値は任意の型を持ちうるということです。低レベルのシステム操作を使わないカスタムコードでこれを使う理由はありません。

```haskell
throw :: Exception e => e -> a
catch :: Exception e => IO a -> (e -> IO a) -> IO a
try :: Exception e => IO a -> IO (Either e a)
evaluate :: a -> IO a
```

```haskell
{-# LANGUAGE DeriveDataTypeable #-}

import Data.Typeable
import Control.Exception

data MyException = MyException
    deriving (Show, Typeable)

instance Exception MyException

evil :: [Int]
evil = [throw MyException]

example1 :: Int
example1 = head evil

example2 :: Int
example2 = length evil

main :: IO ()
main = do
  a <- try (evaluate example1) :: IO (Either MyException Int)
  print a

  b <- try (return example2) :: IO (Either MyException Int)
  print b
```

値は必要でなければ評価されないので、例外は例外が捕捉されるかどうか確証が欲しければ、catch を呼び出す前に標準形へと深く評価してもよいでしょう。``strictCatch`` は標準ライブラリでは提供されていませんが、``deepseq`` を使えば単純に実装できます。

```haskell
strictCatch :: (NFData a, Exception e) => IO a -> (e -> IO a) -> IO a
strictCatch = catch . (toNF =<<)
```

## 例外

先ほどのアプローチの問題は、基本操作を捕捉するのに IO の内部の GHC の非同期例外処理に頼らねばならないことです。``exceptions`` ライブラリは ``Control.Exception`` と同じ API を提供していますが、IO への依存度が低くなっています。

```haskell
{-# LANGUAGE DeriveDataTypeable #-}

import Data.Typeable
import Control.Monad.Catch
import Control.Monad.Identity

data MyException = MyException
    deriving (Show, Typeable)

instance Exception MyException

example :: MonadCatch m => Int -> Int -> m Int
example x y | y == 0 = throwM MyException
            | otherwise = return $ x `div` y

pure :: MonadCatch m => m (Either MyException Int)
pure = do
  a <- try (example 1 2)
  b <- try (example 1 0)
  return (a >> b)
```

参照：

* [exceptions](http://hackage.haskell.org/package/exceptions)

## Either

Either のモナドインスタンスは単純です。束縛 (bind) で Left を偏重していることに注目してください。

```haskell
instance Monad (Either e) where
  return x = Right x

  (Left x)  >>= f = Left x
  (Right x) >>= f = f x
```

いつも見かけるバカバカしい例は、ゼロによる割り算が生じた時に Left の値で失敗し、それ以外の時に Right の値で結果を持つ、安全な割り算の関数を書くものです。

```haskell
sdiv :: Double -> Double -> Either String Double
sdiv _ 0 = throwError "divide by zero"
sdiv i j = return $ i / j

example :: Double -> Double -> Either String Double
example n m = do
  a <- sdiv n m
  b <- sdiv 2 a
  c <- sdiv 2 b
  return c

throwError :: String -> Either String b
throwError a = Left a

main :: IO ()
main = do
  print $ example 1 5
  print $ example 1 0
```

これがかなり馬鹿だということは認めざるを得ませんが、Either や EitherT が例外捕捉に適したモナドである理由の本質を確かに掴んでいます。

## ErrorT

モナド変換子のスタイルでは、``ErrorT`` 変換子を Identity モナドと合成して、``Either Exception a`` へと展開するようにして、使うことができます。この方法は単純ですが、カスタムのException の型が欲しい場合、Exception（あるいは Typeable）型クラスを手動でインスタンス化する必要があります。

```haskell
import Control.Monad.Error
import Control.Monad.Identity

data Exception
  = Failure String
  | GenericFailure
  deriving Show

instance Error Exception where
  noMsg = GenericFailure

type ErrMonad a = ErrorT Exception Identity a

example :: Int -> Int -> ErrMonad Int
example x y = do
  case y of
    0 -> throwError $ Failure "division by zero"
    x -> return $ x `div` y

runFail :: ErrMonad a -> Either Exception a
runFail = runIdentity . runErrorT

example1 :: Either Exception Int
example1 = runFail $ example 2 3

example2 :: Either Exception Int
example2 = runFail $ example 2 0
```

## ExceptT

mtl 2.2 以降では、``ErrorT`` クラスに代わって ``ExceptT`` クラスを使うようになりました。このクラスは、古いクラスの問題の多くを修正しています。

変換子のレベルでは：

```haskell
newtype ExceptT e m a = ExceptT (m (Either e a))

runExceptT :: ExceptT e m a -> m (Either e a)
runExceptT (ExceptT m) = m

instance (Monad m) => Monad (ExceptT e m) where
    return a = ExceptT $ return (Right a)
    m >>= k = ExceptT $ do
        a <- runExceptT m
        case a of
            Left e -> return (Left e)
            Right x -> runExceptT (k x)
    fail = ExceptT . fail

throwE :: (Monad m) => e -> ExceptT e m a
throwE = ExceptT . return . Left

catchE :: (Monad m) =>
    ExceptT e m a               -- ^ 内部の計算
    -> (e -> ExceptT e' m a)    -- ^ 内部の計算の例外のハンドラ
    -> ExceptT e' m a
m `catchE` h = ExceptT $ do
    a <- runExceptT m
    case a of
        Left  l -> runExceptT (h l)
        Right r -> return (Right r)
```

MTL のレベルでは：

```haskell
instance MonadTrans (ExceptT e) where
    lift = ExceptT . liftM Right

class (Monad m) => MonadError e m | m -> e where
    throwError :: e -> m a
    catchError :: m a -> (e -> m a) -> m a

instance MonadError IOException IO where
    throwError = ioError
    catchError = catch

instance MonadError e (Either e) where
    throwError             = Left
    Left  l `catchError` h = h l
    Right r `catchError` _ = Right r
```

参照：

* [Control.Monad.Except](https://hackage.haskell.org/package/mtl-2.2.1/docs/Control-Monad-Except.html)

## EitherT

```haskell
newtype EitherT e m a = EitherT {runEitherT :: m (Either e a)}
        -- Defined in `Control.Monad.Trans.Either'
```

```haskell
runEitherT :: EitherT e m a -> m (Either e a)
tryIO :: MonadIO m => IO a -> EitherT IOException m a

throwT  :: Monad m => e -> EitherT e m r
catchT  :: Monad m => EitherT a m r -> (a -> EitherT b m r) -> EitherT b m r
handleT :: Monad m => (a -> EitherT b m r) -> EitherT a m r -> EitherT b m
```

使うべき理想のモナドは単純に ``EitherT`` のモナドであり、これを ``ErrorT`` と似た API で使えることを望んでいます。例えば、``read`` を使って標準入力の正の整数を読もうとしているとしましょう。2 つの失敗のモードと 2 つの失敗のケースがあります。一方は、``Prelude.readIO`` からのエラーで失敗する、パースエラーに対するものであり、一方は、検査によるカスタムの例外で失敗する、非正整数に対するものです。同じ変換子の中で 2 つのケースを統一して扱いたいものです。

``safe`` ライブラリと ``errors`` ライブラリを組み合わせて使えば、``EitherT`` を使う暮らしはより楽なものになります。safe ライブラリは、失敗を、Maybe の値、明示的に渡されたデフォルト値、あるいはより情報の多い”注釈”という例外として扱う、標準の Prelude の関数のより安全な変種をいろいろと提供しています。一方で、errors ライブラリは安全な Maybe の関数を再エクスポートし、それらの関数を ``EitherT`` モナドへと持ち上げるために、``try`` という接頭辞を持つ関数のグループを提供しています。これらの関数は、アクションを実行し、例外で失敗する可能性があります。

```haskell
-- `read` と等価な例外処理
tryRead :: (Monad m, Read a) => e -> String -> EitherT e m a

-- `head` と等価な例外処理
tryHead :: Monad m => e -> [a] -> EitherT e m a

-- `(!!)` と等価な例外処理
tryAt :: Monad m => e -> [a] -> Int -> EitherT e m a
```

```haskell
import Control.Error
import Control.Monad.Trans

data Failure
  = NonPositive Int
  | ReadError String
  deriving Show

main :: IO ()
main = do
  putStrLn "正の数を入力してください"
  s <- getLine

  e <- runEitherT $ do
      n <- tryRead (ReadError s) s
      if n > 0
        then return $ n + 1
        else throwT $ NonPositive n

  case e of
      Left  e -> putStrLn $ "この例外で失敗：" ++ show e
      Right n -> putStrLn $ "この値で成功：" ++ show n
```

参照：

* [Error Handling Simplified](http://www.haskellforall.com/2012/07/errors-10-simplified-error-handling.html)
* [Safe](http://hackage.haskell.org/package/safe)
