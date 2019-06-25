[Prerequisites]
1. .NET Core 2.1 for your Operation System
2. CUDA SDK (Optional - if you want calculations to be much faster)

[Picture change]
Current picture path is now hardcoded, so, to use another picture, you need to change "picturePath" variable in DatasetWrapper class 
	to path to your desired hyperspectral image.

[Run]
Run with Administrative (Root) privileges from project folder: 
	dotnet run --framework=netcoreapp2.0

[Using]
Render band (and band number) - render selected band of HSI
Render band Canny - rendering canny for selected band (or current picture)
Save picture - do as it's sound
Render histogram - show historgam and save details in .csv file
Render brightness calculation data - render calculations for mean, max, and standart deviation of brightness for each band, save results in .csv
---[For next function, you should use brightness calculation atleast once]
Render byte representation() - calculating byte representation for HSI.
---[For next functions, you should use byte representation atleast once]
Edge via signature length +(normalized) +(byte)
Accumulate edges



Render correlation map, Render sobel map, render scanline - unsupported features, used for testing purpose only.