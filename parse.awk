BEGIN {
  total = 0
}

{
  total += $0
  print "("NR"," total")"
}