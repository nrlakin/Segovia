/*
 *	PsThreadVerify.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import com.fujitsu.frontech.palmsecure_smpl.data.PsDataManager;
import com.fujitsu.frontech.palmsecure_smpl.data.PsThreadResult;
import com.fujitsu.frontech.palmsecure_smpl.event.PsBusinessListener;
import com.fujitsu.frontech.palmsecure_smpl.exception.PsAplException;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorIni;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorLang;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;

public class PsThreadVerify extends PsThreadBase {

	public PsThreadVerify(PsMainFrame frame, PsBusinessListener businesslistener, PalmSecureIf palmsecureIf, JAVA_uint32 moduleHandle, String userID) {
		super(frame, businesslistener, palmsecureIf, moduleHandle, userID);
	}

	public void run() {

		PsThreadResult stResult = new PsThreadResult();

		try {
			PsFileAccessorIni iniAcs = PsFileAccessorIni.GetInstance();
			PsDataManager dataMng = PsDataManager.GetInstance();

			int waitTime = 0;
			int numOfRetry = iniAcs.GetValueInteger(PsFileAccessorIni.NumberOfRetry);
			stResult.userId.add(userID);

			byte[] bufferBIR = null;

			try {
				bufferBIR = dataMng.Ps_Sample_Apl_Java_GetTemplate(userID);
			} catch(PsAplException e) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e.getErrorMsgKey());
			}

			//Get a instance of DNET_BioAPI_INPUT_BIR class
			///////////////////////////////////////////////////////////////////////////
			JAVA_BioAPI_INPUT_BIR storedTemplate = new JAVA_BioAPI_INPUT_BIR();
			storedTemplate.Form = PalmSecureConstant.JAVA_BioAPI_FULLBIR_INPUT;
			try {
				storedTemplate.BIR = PalmSecureHelper.convertByteToBIR(bufferBIR);
			} catch(PalmSecureException e) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
			}
			///////////////////////////////////////////////////////////////////////////

			//Repeat numOfRetry times until verification succeed
			int verifyCnt = 0;
			for (verifyCnt = 0; verifyCnt <= numOfRetry; verifyCnt++) {

				Ps_Sample_Apl_Java_NotifyWorkMessage(
						PsFileAccessorLang.Guidance_WorkVerify,
						verifyCnt + 1);

				if (verifyCnt > 0) {

					Ps_Sample_Apl_Java_NotifyGuidance(
							PsFileAccessorLang.Guidance_RetryTransaction,
							false);

					waitTime = 0;

					do {
						//End transaction in case of cancel
						if (this.frame.cancelFlg == true) {
							break;
						}
						if (waitTime < iniAcs.GetValueInteger(PsFileAccessorIni.SleepTime)) {
							Thread.sleep(100);
							waitTime = waitTime + 100;
						} else {
							break;
						}
					}while(true);
				}

				//End transaction in case of cancel
				if (this.frame.cancelFlg == true) {
					break;
				}

				stResult.retryCnt = verifyCnt;

				//Verification
				///////////////////////////////////////////////////////////////////////////
				JAVA_sint32 maxFRRRequested = new JAVA_sint32();
				maxFRRRequested.value = PalmSecureConstant.JAVA_BioAPI_FALSE;
				JAVA_uint32 farPrecedence = new JAVA_uint32();
				farPrecedence.value = PalmSecureConstant.JAVA_BioAPI_FALSE;
				JAVA_uint32 result = new JAVA_uint32();
				JAVA_sint32 farAchieved = new JAVA_sint32();
				JAVA_sint32 timeout = new JAVA_sint32();
				try {
					stResult.result = palmsecureIf.JAVA_BioAPI_Verify(
							moduleHandle,
							null,
							maxFRRRequested,
							null,
							storedTemplate,
							null,
							result,
							farAchieved,
							null,
							null,
							timeout,
							null);
				} catch(PalmSecureException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
					break;
				}
				///////////////////////////////////////////////////////////////////////////

				//End transaction in case of cancel
				if (this.frame.cancelFlg == true) {
					break;
				}

				//If PalmSecure method failed, get error info
				if (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK) {
					try {
						palmsecureIf.JAVA_PvAPI_GetErrorInfo(stResult.errInfo);
						PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, stResult.errInfo);
					} catch(PalmSecureException e) {
					}
					break;
				}

				//Log a shilouette image
				stResult.info = this.frame.silhouette;

				//If result of verification is false, retry verification
				if (result.value != PalmSecureConstant.JAVA_BioAPI_TRUE) {
					continue;
				}

				stResult.authenticated = true;
				break;

			}

			Ps_Sample_Apl_Java_NotifyResult_Verify(stResult);

		} catch(Exception e) {
		}

		return;

	}


}
