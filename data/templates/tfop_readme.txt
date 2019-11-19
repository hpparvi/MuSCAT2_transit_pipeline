===================
MuSCAT2 TFOP README
===================

Frame
------
The "*frame*.pdf" figures show a MuSCAT2 reference frame with the star IDs, stars in the GAIA catalog, etc. in three
different magnifications. Stars chosen for photometry are marked with dashed circles with a number indicating the star
ID (target = 0 and rest are running from brightest to faintest). Stars within 2 arcmin from the target that are bright
enough to cause the transit candidate event are marked with an additional solid circle, and stars withing 2 arcmin from
the target but too faint to cause the event are marked with dotted circles (the circle radii depend on the field
crowdedness).


Transit fit
---------
This "*transit_fit.pdf" figure gives an overview to transit modelling results and lists the probabilities that the
transit occurs during the observing window.

The transit probabilities are calculated based on the TFOP transit centre and orbital period estimates with their
uncertainties. The listed probabilities are

P(full transit in window): the probability that the transit is fully in the observing window
P(ingress): the probability that the ingress occurs inside the observing window
P(egress): the probability that the egress occurs inside the observing window
P(misses window): the probability that the transit occurs completely outside the observing window
P(spans window): the probability that the transit begins before and ends after the observing window

The "Raw light curve, model, and residuals" panel plots the data and the fitted model in different ways. The first row
shows the residual distribution (as a histogram) and the fitted error distribution with its width; the second row  shows
the absolute raw target light curve and the fitted flux model with trends from covariates and the comparison stars; the
third row shows the detrended relative target light curve and the fitted transit model; and the last row shows the
residuals.

The blue vertical line spanning most of the vertical extent of the panels indicates the expected transit centre and the
blue shaded regions show its 1, 2, and 3 sigma uncertainties. Same information is shown for the ingress and egress at
the upper edge of each panel. The dotted blue horizontal line shows the expected transit depth.

Raw LCS
---------

The "*raw_lcs.pdf" plot shows the absolute raw photometry for the target star (id=0 unless otherwise stated) and stars
within 2 arcmin distance from it (possible blends). The star/target flux ratio is written at the upper right corner of
each plot, and a transit model is plotted at the expected transit location with the expected transit duration and a depth
that would be required to create the observed transit candidate signal if the observed flux would comprise only of the
target and possible blend (that is, we ignore the rest of the stars and compare only one blend and the target at a time.
In reality the signal would need to be greater if the light curve contains significant contamination from multiple sources.)

Covariates
-----------
The "*covariates.pdf" plots show the covariates measured simultaneously with the photometry exposures (or estimated from
the frames.)


