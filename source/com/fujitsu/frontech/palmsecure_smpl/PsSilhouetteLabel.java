/*
 *	PsSilhouetteLabel.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JLabel;

public class PsSilhouetteLabel extends JLabel {

	private static final long serialVersionUID = 1L;

	private BufferedImage displayedImage = null;
	private BufferedImage handguideImage = null;

	protected PsSilhouetteLabel() {
	}

	public void paint(Graphics g) {
		super.paint(g);
		g.drawImage(displayedImage, 0, 0, null);
	}

	public void Ps_Sample_Apl_Java_SetHandGuide(BufferedImage handGuide) {

		if (handguideImage == null) {

			int displayW = getWidth();
			int displayH = getHeight();

			BufferedImage resize = new BufferedImage(displayW, displayH, handGuide.getType());
			Graphics g = resize.getGraphics();
			g.drawImage(
					handGuide.getScaledInstance(displayW, displayH, Image.SCALE_SMOOTH),
					0, 0, displayW, displayH, null);
			g.dispose();

			handguideImage = resize;
		}

		displayedImage = handguideImage;

	}

	public void Ps_Sample_Apl_Java_SetSilhouette(byte[] silhouetteImage) throws IOException {

		if ((silhouetteImage != null) && (0 < silhouetteImage.length)) {
			ByteArrayInputStream bais = new ByteArrayInputStream(silhouetteImage);
			BufferedImage orgImage = ImageIO.read(bais);

			double displayW = getWidth();
			double displayH = getHeight();

			double imageW = orgImage.getWidth();
			double imageH = orgImage.getHeight();

			try {

				BufferedImage resize = new BufferedImage((int)displayW, (int)displayH, orgImage.getType());

				AffineTransform affine = new AffineTransform(-displayW/imageW, 0, 0, displayH/imageH, displayW, 0);
				AffineTransformOp affineOp = new AffineTransformOp(affine, null);
				affineOp.filter(orgImage, resize);

				displayedImage = resize;

			} catch(Exception e) {

			}

		}
	}
}
