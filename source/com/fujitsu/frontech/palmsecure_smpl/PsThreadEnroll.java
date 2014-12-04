/*
 * PsThreadEnroll.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import java.util.ArrayList;

import com.fujitsu.frontech.palmsecure_smpl.data.PsDataManager;
import com.fujitsu.frontech.palmsecure_smpl.data.PsThreadResult;
import com.fujitsu.frontech.palmsecure_smpl.event.PsBusinessListener;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorIni;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorLang;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;

public class PsThreadEnroll extends PsThreadBase {

	public PsThreadEnroll(PsMainFrame frame, PsBusinessListener businesslistener, PalmSecureIf palmsecureIf, JAVA_uint32 moduleHandle, String userID) {
		super(frame, businesslistener, palmsecureIf, moduleHandle, userID);
	}

	public void run() {

		PsThreadResult stResult = new PsThreadResult();
		PsFileAccessorIni iniAcs = PsFileAccessorIni.GetInstance();

		try {

			int enrollCnt = 0;
			int enrollScore = 0;
			int waitTime = 0;
			int numOfRetry = iniAcs.GetValueInteger(PsFileAccessorIni.NumberOfRetry);

			ArrayList<Integer> scoreList = new ArrayList<Integer>();
			for (enrollCnt = 0; enrollCnt <= numOfRetry; enrollCnt++) {

				if (enrollCnt > 0) {
					Ps_Sample_Apl_Java_NotifyGuidance(PsFileAccessorLang.Guidance_RetryTransaction, false);

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

				this.frame.enrollFlg = true;
				stResult.retryCnt = enrollCnt;

				//Enrollment
				///////////////////////////////////////////////////////////////////////////
				JAVA_uint8 purpose = new JAVA_uint8();
				purpose.value = PalmSecureConstant.JAVA_BioAPI_PURPOSE_VERIFY;
				JAVA_sint32 birHandle = new JAVA_sint32();
				JAVA_sint32 timeout = new JAVA_sint32();
				try {
					stResult.result = palmsecureIf.JAVA_BioAPI_Enroll(
							moduleHandle,
							purpose,
							null,
							birHandle,
							null,
							timeout,
							null);
				} catch(PalmSecureException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
					stResult.result = PalmSecureConstant.JAVA_BioAPI_ERRCODE_FUNCTION_FAILED;
					this.frame.enrollFlg = false;
					break;
				}
				///////////////////////////////////////////////////////////////////////////

				this.frame.enrollFlg = false;

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

				enrollScore = this.frame.notifiedScore;

				//Log a shilouette image
				stResult.info = this.frame.silhouette;

				//Get BIR data ( vein data )
				///////////////////////////////////////////////////////////////////////////
				JAVA_BioAPI_BIR BIR = new JAVA_BioAPI_BIR();
				try {
					stResult.result = palmsecureIf.JAVA_BioAPI_GetBIRFromHandle(
							moduleHandle,
							birHandle,
							BIR);
				} catch(PalmSecureException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
					stResult.result = PalmSecureConstant.JAVA_BioAPI_ERRCODE_FUNCTION_FAILED;
					break;
				}
				///////////////////////////////////////////////////////////////////////////

				//If PalmSecure method failed, get error info
				if (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK) {
					try {
						palmsecureIf.JAVA_PvAPI_GetErrorInfo(stResult.errInfo);
						PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, stResult.errInfo);
					} catch(PalmSecureException e) {
					}
					break;
				}

				//Free BIR handle
				///////////////////////////////////////////////////////////////////////////
				try {
					stResult.result = palmsecureIf.JAVA_BioAPI_FreeBIRHandle(
							moduleHandle,
							birHandle);
				} catch(PalmSecureException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
					stResult.result = PalmSecureConstant.JAVA_BioAPI_ERRCODE_FUNCTION_FAILED;
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

				//Repeat 2 times until enrollment test failed
				boolean retryFlg = false;
				scoreList.clear();
				for( int cnt = 0; cnt < 2; cnt++ ) {
					if (cnt == 0) {
						Ps_Sample_Apl_Java_NotifyGuidance(
								PsFileAccessorLang.Guidance_EnrollmentTest,
								false);
					} else {
						Ps_Sample_Apl_Java_NotifyGuidance(
								PsFileAccessorLang.Guidance_RetryTransaction,
								false);
					}

					Ps_Sample_Apl_Java_NotifyWorkMessage(
							PsFileAccessorLang.Guidance_WorkEnrollTest,
							cnt+1);
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

					//End transaction in case of cancel
					if (this.frame.cancelFlg == true) {
						break;
					}

					//Set mode to get authentication score
					///////////////////////////////////////////////////////////////////////////
					JAVA_uint32 dwFlag = new JAVA_uint32();
					dwFlag.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS;
					JAVA_uint32 dwParam1 = new JAVA_uint32();
					dwParam1.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS_ON;
					try {
						stResult.result = palmsecureIf.JAVA_PvAPI_SetProfile(
								moduleHandle,
								dwFlag,
								dwParam1,
								null,
								null);
					} catch(PalmSecureException e) {
						PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
						stResult.result = PalmSecureConstant.JAVA_BioAPI_ERRCODE_FUNCTION_FAILED;
						break;
					}
					///////////////////////////////////////////////////////////////////////////

					//If PalmSecure method failed, get error info.
					if (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK) {
						try {
							palmsecureIf.JAVA_PvAPI_GetErrorInfo(stResult.errInfo);
							PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, stResult.errInfo);
						} catch(PalmSecureException e) {
						}
						break;
					}

					//Verification to check template quality
					///////////////////////////////////////////////////////////////////////////
					JAVA_sint32 maxFRRRequested = new JAVA_sint32();
					maxFRRRequested.value = PalmSecureConstant.JAVA_PvAPI_MATCHING_LEVEL_NORMAL;
					JAVA_uint32 farPrecedence = new JAVA_uint32();
					farPrecedence.value = PalmSecureConstant.JAVA_BioAPI_FALSE;
					JAVA_BioAPI_INPUT_BIR storedTemplate = new JAVA_BioAPI_INPUT_BIR();
					storedTemplate.Form = PalmSecureConstant.JAVA_BioAPI_FULLBIR_INPUT;
					storedTemplate.BIR = BIR;
					JAVA_uint32 result = new JAVA_uint32();
					JAVA_sint32 farAchieved = new JAVA_sint32();
					timeout.value = 0;
					try {
						stResult.result = palmsecureIf.JAVA_BioAPI_Verify(
								moduleHandle,
								null,
								maxFRRRequested,
								farPrecedence,
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
						stResult.result = PalmSecureConstant.JAVA_BioAPI_ERRCODE_FUNCTION_FAILED;
						break;
					}
					///////////////////////////////////////////////////////////////////////////

					//If PalmSecure method failed, get error info
					if ( (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK)
							&& this.frame.cancelFlg != true) {
						try {
							palmsecureIf.JAVA_PvAPI_GetErrorInfo(stResult.errInfo);
							PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, stResult.errInfo);
						} catch(PalmSecureException e) {
						}
						break;
					}

					//Set mode not to get authentication score
					///////////////////////////////////////////////////////////////////////////
					dwFlag.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS;
					dwParam1.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS_OFF;
					try {
						stResult.result = palmsecureIf.JAVA_PvAPI_SetProfile(
								moduleHandle,
								dwFlag,
								dwParam1,
								null,
								null);
					} catch(PalmSecureException e) {
						PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
						stResult.result = PalmSecureConstant.JAVA_BioAPI_ERRCODE_FUNCTION_FAILED;
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

					//If result of verification is false, retry enrollment test
					if (result.value != PalmSecureConstant.JAVA_BioAPI_TRUE) {
						stResult.authenticated = false;
						retryFlg = true;
						break;
					}

					stResult.authenticated = true;
					scoreList.add(farAchieved.value);

				}

				if (this.frame.cancelFlg == true) {
					break;
				}
				if (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK) {
					break;
				}
				if (retryFlg == true) {
					continue;
				}

				//Get an average score
				int veriScore = 0;
				for (int i = 0; i < scoreList.size(); i++) {
					veriScore = veriScore + scoreList.get(i);
				}
				if (scoreList.size() > 0) {
					veriScore = veriScore / scoreList.size();
				}
				stResult.farAchieved.add(veriScore);

				byte[] bufferBIR = null;

				//Create a byte array of vein data and output vein data to file
				///////////////////////////////////////////////////////////////////////////
				try {
					bufferBIR = PalmSecureHelper.convertBIRToByte(BIR);
				} catch(PalmSecureException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
					stResult.result = PalmSecureConstant.JAVA_BioAPI_ERRCODE_FUNCTION_FAILED;
					break;
				}
				///////////////////////////////////////////////////////////////////////////

				PsDataManager dataMng = PsDataManager.GetInstance();
				dataMng.Ps_Sample_Apl_Java_Insert(userID, bufferBIR);

				break;

			}

			Ps_Sample_Apl_Java_NotifyResult_Enroll(stResult, enrollScore);

		} catch(Exception e) {
		}

		return;

	}

}
