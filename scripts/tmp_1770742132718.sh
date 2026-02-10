mkdir -p ~/Desktop/programming && cd ~/Desktop/programming
for lang in python ruby java c c++; do
  echo -e '#!/bin/bash\necho -e "Hello World"'> $lang.sh
  case $lang in
    python) echo "print('Hello World')" >> $lang.sh ;;
    ruby)  echo "puts 'Hello World'" >> $lang.sh ;;
    java)  echo "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello World\"); } }" > HelloWorld.java; javac HelloWorld.java ;; 
    c)     echo "#include <stdio.h>\nint main() { printf(\"Hello World\\n\"); return 0; }" > $lang.c ;;
    c++)   echo "#include <stdio.h>\nint main() { printf(\"Hello World\\n\"); return 0; }" > $lang.cpp ;;
  esac
  chmod +x $lang.sh
done
echo 'Documents created'