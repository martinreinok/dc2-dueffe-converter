#

## VRP
```
V201= real size x
V202= design size x
V203= real size y
V204= design size y
V205= origin x
V206= orign y
V207= partial length
V208= total length
V209=10
```

## CNC Commands

### Starting Block, always similar
```
BLOCK VA1
VELL 166.67
ACCL 333.33
w195=1
ENDBL
BLOCK VA2
VELL 133.33
ACCL 166.67
w195=2
ENDBL
BLOCK VA3
VELL 100
ACCL 133.33
w195=3
ENDBL
;
PROGRAM LAV1
; DESIGN: DESIGN NAME
ABS X=0
ABS Y=0
CORNER a=333.33
VEL X= 80
VEL Y= 80
ACC X= 30
ACC Y= 30
v990=1
v991=1
CALL INIZIO
LABEL 1
CALL INLAV1
CALL VA1
```