@echo off
set CLASSPATH=.;f3bc4jav.jar;%CLASSPATH%

@echo on
javac com\fujitsu\frontech\palmsecure_smpl\data\PsDataManager.java
javac com\fujitsu\frontech\palmsecure_smpl\data\PsLogManager.java
javac com\fujitsu\frontech\palmsecure_smpl\data\PsThreadResult.java
javac com\fujitsu\frontech\palmsecure_smpl\data\PsVeinDataFilenameFilter.java
javac com\fujitsu\frontech\palmsecure_smpl\event\PsBusinessListener.java
javac com\fujitsu\frontech\palmsecure_smpl\exception\PsAplException.java
javac com\fujitsu\frontech\palmsecure_smpl\xml\PsFileAccessor.java
javac com\fujitsu\frontech\palmsecure_smpl\xml\PsFileAccessorIni.java
javac com\fujitsu\frontech\palmsecure_smpl\xml\PsFileAccessorLang.java
javac com\fujitsu\frontech\palmsecure_smpl\PsInputIdText.java
javac com\fujitsu\frontech\palmsecure_smpl\PsMainFrame.java
javac com\fujitsu\frontech\palmsecure_smpl\PsMessageDialog.java
javac com\fujitsu\frontech\palmsecure_smpl\PsSampleApl.java
javac com\fujitsu\frontech\palmsecure_smpl\PsSilhouetteLabel.java
javac com\fujitsu\frontech\palmsecure_smpl\PsStateCallback.java
javac com\fujitsu\frontech\palmsecure_smpl\PsStreamingCallback.java
javac com\fujitsu\frontech\palmsecure_smpl\PsThreadBase.java
javac com\fujitsu\frontech\palmsecure_smpl\PsThreadCancel.java
javac com\fujitsu\frontech\palmsecure_smpl\PsThreadEnroll.java
javac com\fujitsu\frontech\palmsecure_smpl\PsThreadIdentify.java
javac com\fujitsu\frontech\palmsecure_smpl\PsThreadVerify.java

@echo off
md temp\com\fujitsu\frontech\palmsecure_smpl
xcopy com\fujitsu\frontech\palmsecure_smpl\*.class temp\com\fujitsu\frontech\palmsecure_smpl /s /q /y

pushd com\fujitsu\frontech\palmsecure_smpl
del /s /q *.class
popd

pushd temp
@echo on
jar -cf PalmSecureSample_Java.jar com
@echo off
popd

move /y temp\PalmSecureSample_Java.jar PalmSecureSample_Java.jar
rd /s /q temp
