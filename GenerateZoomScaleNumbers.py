currScale = 1
scaleMultiplier = 1.2
framesLeft = 60

out = ""

while framesLeft > 0:
  out += str(currScale) + "\n"
  currScale *= scaleMultiplier
  framesLeft -= 1

print(out)
