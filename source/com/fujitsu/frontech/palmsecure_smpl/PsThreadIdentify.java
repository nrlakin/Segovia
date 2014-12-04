/*
 *	PsThreadIdentify.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import java.util.ArrayList;

import com.fujitsu.frontech.palmsecure_smpl.data.PsDataManager;
import com.fujitsu.frontech.palmsecure_smpl.data.PsThreadResult;
import com.fujitsu.frontech.palmsecure_smpl.event.PsBusinessListener;
import com.fujitsu.frontech.palmsecure_smpl.exception.PsAplException;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorIni;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorLang;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;

public class PsThreadIdentify extends PsThreadBase {

	public PsThreadIdentify(PsMainFrame frame, PsBusinessListener businesslistener, PalmSecureIf palmsecureIf, JAVA_uint32 moduleHandle) {

		super(frame, businesslistener, palmsecureIf, moduleHandle, "");
	}

	public void run() {

		PsThreadResult stResult = null;
		PsDataManager dataMng = PsDataManager.GetInstance();
		PsFileAccessorIni iniAcs = PsFileAccessorIni.GetInstance();

		try {
			int waitTime = 0;
			int numOfRetry = iniAcs.GetValueInteger(PsFileAccessorIni.NumberOfRetry);

			ArrayList<String> idList = dataMng.Ps_Sample_Apl_Java_GetAllUserId();
			int memberNum = idList.size();

			//Create an array of DNET_BioAPI_BIR class
			///////////////////////////////////////////////////////////////////////////
			JAVA_BioAPI_BIR[] birAry = new JAVA_BioAPI_BIR[memberNum];
			///////////////////////////////////////////////////////////////////////////

			//Read vein data from a file
			for (int i = 0; i < memberNum; i++) {

				byte[] bufferBIR = null;

				try {
					bufferBIR = dataMng.Ps_Sample_Apl_Java_GetTemplate(idList.get(i));
				} catch(PsAplException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e.getErrorMsgKey());
				}

				//Get a instance of DNET_BioAPI_BIR class
				///////////////////////////////////////////////////////////////////////////
				try {
					birAry[i] = PalmSecureHelper.convertByteToBIR(bufferBIR);
				} catch(PalmSecureException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
					break;
				}
				///////////////////////////////////////////////////////////////////////////
			}

			//Create a instance of DNET_BioAPI_IDENTIFY_POPULATION class
			///////////////////////////////////////////////////////////////////////////
			JAVA_BioAPI_BIR_ARRAY_POPULATION BIRAryPopu = new JAVA_BioAPI_BIR_ARRAY_POPULATION();
			BIRAryPopu.NumberOfMembers = memberNum;
			BIRAryPopu.Members = birAry;
			JAVA_BioAPI_IDENTIFY_POPULATION Population = new JAVA_BioAPI_IDENTIFY_POPULATION();
			Population.Type = PalmSecureConstant.JAVA_BioAPI_ARRAY_TYPE;
			Population.BIRArray = BIRAryPopu;
			///////////////////////////////////////////////////////////////////////////

			//Repeat numOfRetry times until identification succeed
			for (int identifyCnt = 0; identifyCnt <= numOfRetry; identifyCnt++) {

				Ps_Sample_Apl_Java_NotifyWorkMessage(
						PsFileAccessorLang.Guidance_WorkIdentify,
						identifyCnt + 1);

				if (identifyCnt > 0) {

					Ps_Sample_Apl_Java_NotifyGuidance(
							PsFileAccessorLang.Guidance_RetryTransaction,
							false);

					waitTime = 0;
					do {
						//End transaction in case of cancel
						if (this.frame.cancelFlg == true) {
							break;
						}

						if (waitTime < iniAcs.GetValueInteger(PsFileAccessorIni.SleepTime) ) {
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

				stResult = new PsThreadResult();

				//Set mode to get authentication score
				///////////////////////////////////////////////////////////////////////////
				try {
					JAVA_uint32 dwFlag = new JAVA_uint32();
					dwFlag.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS;
					JAVA_uint32 dwParam1 = new JAVA_uint32();
					dwParam1.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS_ON;
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

				//If PalmSecure method failed, get error info
				if (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK) {
					try {
						palmsecureIf.JAVA_PvAPI_GetErrorInfo(stResult.errInfo);
						PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, stResult.errInfo);
					} catch(PalmSecureException e) {
					}
				}

				stResult.retryCnt = identifyCnt;

				//Identification
				///////////////////////////////////////////////////////////////////////////
				JAVA_sint32 maxFRRRequested = new JAVA_sint32();
				maxFRRRequested.value = PalmSecureConstant.JAVA_PvAPI_MATCHING_LEVEL_NORMAL;
				JAVA_uint32 farPrecedence = new JAVA_uint32();
				farPrecedence.value = PalmSecureConstant.JAVA_BioAPI_FALSE;
				JAVA_uint32 binning = new JAVA_uint32();
				binning.value = PalmSecureConstant.JAVA_BioAPI_FALSE;
				JAVA_uint32 maxNumberOfResults = new JAVA_uint32();
				maxNumberOfResults.value = iniAcs.GetValueInteger(PsFileAccessorIni.MaxResults);
				JAVA_uint32 numberOfResults = new JAVA_uint32();
				JAVA_BioAPI_CANDIDATE[] candidates = new JAVA_BioAPI_CANDIDATE[(int)maxNumberOfResults.value];
				JAVA_sint32 timeout = new JAVA_sint32();
				timeout.value = 0;
				try {
					stResult.result = palmsecureIf.JAVA_BioAPI_Identify(
							moduleHandle,
							null,
							maxFRRRequested,
							farPrecedence,
							Population,
							binning,
							maxNumberOfResults,
							numberOfResults,
							candidates,
							timeout,
							null);
				} catch(PalmSecureException e) {
					PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, e);
					break;
				}
				///////////////////////////////////////////////////////////////////////////

				//If PalmSecure method failed, get error info
				if ( (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK)
						&& (this.frame.cancelFlg != true)) {
					try {
						palmsecureIf.JAVA_PvAPI_GetErrorInfo(stResult.errInfo);
						PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this.frame, stResult.errInfo);
					} catch(PalmSecureException e) {
					}
					break;
				}

				//Set mode not to get authentication score
				///////////////////////////////////////////////////////////////////////////
				try {
					JAVA_uint32 dwFlag = new JAVA_uint32();
					dwFlag.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS;
					JAVA_uint32 dwParam1 = new JAVA_uint32();
					dwParam1.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_SCORE_NOTIFICATIONS_OFF;
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

				//Log a shilouette image
				stResult.info = this.frame.silhouette;

				//If result of identification is 0, retry identification
				if (numberOfResults.value == 0) {
					continue;
				}
				if (numberOfResults.value >= 1) {
					for (int i = 0; i < numberOfResults.value; i++) {
						stResult.farAchieved.add((int)candidates[i].FARAchieved);
						stResult.userId.add(idList.get((int)(candidates[i].BIRInArray)));
					}
					if (numberOfResults.value > 1 ) {
						long mathWork1 = candidates[0].FARAchieved;
						long mathWork2 = candidates[1].FARAchieved;
						if ( (mathWork1 - mathWork2) < 3000) {
							continue;
						}
					}
				}

				stResult.authenticated = true;

				break;
			}

			Ps_Sample_Apl_Java_NotifyResult_Identify(stResult);

		} catch(Exception e) {
		}

		return;

	}
}
