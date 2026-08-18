package d2l

import scala.quoted.*

/** A replacement for pprint's type printer, for use from notebook cells.
  *
  * Ammonite renders each `name: Type = value` line by summoning a
  * `pprint.TPrint[T]` at the call site of `ReplBridge.value.Internal.print`,
  * i.e. inside the cell's own wrapper.  pprint's instance lives in
  * `object TPrint extends TPrintLowPri`, so anything imported into cell scope
  * takes precedence over it -- which is what makes this file possible.
  *
  * Why it exists: pprint's instance walks the type structurally, and the
  * fallback branch of that walker (`pprint/TPrintImpl.scala`,
  * `case _ => Type.show[T]`) prints the *top level* type rather than the node
  * it is currently visiting.  Every node it has no case for is therefore
  * replaced by a copy of the whole value's type, so with dimwit
  * `t0.pow(t0)` renders as `Tensor[Tensor[EmptyTuple, Float32], Float32]`,
  * and inside a tuple the substitution repeats once per element.
  *
  * The walker below follows pprint's (MIT, com-lihaoyi/PPrint) so that tuples
  * and functions keep their familiar sugar, with two changes: the fallback
  * prints the node it is on, and a `*:` chain ending in `EmptyTuple` is printed
  * as a tuple -- shapes arrive in cons form, and `Tensor[(Row, Col), Float32]`
  * reads better than `Tensor[Row *: Col *: EmptyTuple, Float32]`.
  *
  * This cannot live in a notebook cell: Ammonite wraps cell code in a class,
  * so a macro implementation declared there is not static and the compiler
  * rejects the splice.  Hence a published jar.
  */
object TPrintNice:

  inline given nice[T]: pprint.TPrint[T] = ${ impl[T] }

  def impl[T: Type](using Quotes): Expr[pprint.TPrint[T]] =
    import quotes.reflect.*

    val functionTypes = (0 to 22).map(i => s"scala.Function$i").toSet
    val tupleTypes = (0 to 22).map(i => s"scala.Tuple$i").toSet

    // `A *: B *: EmptyTuple` -> Some(List(A, B)); an open tail -> None.
    def tupleChain(tpe: TypeRepr): Option[List[TypeRepr]] =
      if tpe =:= TypeRepr.of[EmptyTuple] then Some(Nil)
      else
        tpe match
          case AppliedType(tycon, List(head, tail)) if tycon.typeSymbol.fullName == "scala.*:" =>
            tupleChain(tail).map(head :: _)
          case _ => None

    def rec(tpe: TypeRepr): String = tpe match
      case TypeBounds(lo, hi) =>
        val l = if lo =:= TypeRepr.of[Nothing] then "" else s" >: ${rec(lo)}"
        val h = if hi =:= TypeRepr.of[Any] then "" else s" <: ${rec(hi)}"
        s"_$l$h"
      case AppliedType(tycon, args) =>
        val full = tycon.typeSymbol.fullName
        if functionTypes.contains(full) then
          val params = args.init
          val body = rec(args.last)
          if params.sizeIs == 1 then s"${rec(params.head)} => $body"
          else params.map(rec).mkString("(", ", ", ")") + s" => $body"
        else if tupleTypes.contains(full) then args.map(rec).mkString("(", ", ", ")")
        else if full == "scala.*:" then
          tupleChain(tpe) match
            case Some(elems) => elems.map(rec).mkString("(", ", ", ")")
            case None        => args.map(rec).mkString(" *: ")
        else tycon.typeSymbol.name + args.map(rec).mkString("[", ", ", "]")
      case AnnotatedType(parent, _) => rec(parent)
      case TypeRef(_, name)         => name
      // The branch pprint gets wrong: print *this* node, not the outer type.
      case other => other.show(using Printer.TypeReprShortCode)

    val rendered = rec(TypeRepr.of[T])
    '{ pprint.TPrint.recolor[T](fansi.Color.Green(fansi.Str(${ Expr(rendered) }))) }
