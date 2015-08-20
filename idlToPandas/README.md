idlToPandas
===========

`idlToPandas` is a library to import a saved IDL structure and return it as 
a pandas DataFrame. 


###Arguments
**filename**: the filename of the IDL save file  
**keyValue [optional]**: if there is more than one variable stored in the IDL save file `keyValue` specifies which variable is the IDL structure you would like to convert into a pandas DataFrame.


###Example
```
from idlToPandas import *  
rvStruct = idlToPandas('/my/data/dir/vst123456.dat', 
			keyValue='cf3')  

print(rvStruct.OBNM)  
print(rvStruct.keys())  
```

###Dependencies

* argparse
* pandas
* scipy