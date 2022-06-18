---
layout: page
mainnav: true
title: About
mathjax: false
permalink: /about/
---

<!---
change contentid, content url, and metadata name.
-->

### Thomas Chiang's Curriculum Vitae


<div id="adobe-dc-view" style="width: 100%;"></div>
<script src="https://documentcloud.adobe.com/view-sdk/main.js"></script>
<script type="text/javascript">
	document.addEventListener("adobe_dc_view_sdk.ready", function(){ 
		var adobeDCView = new AdobeDC.View({clientId: "3e66a9482764407187a81a0bf601400a", divId: "adobe-dc-view"});
		adobeDCView.previewFile({
			content:{location: {url: "https://tomchiang30115.github.io/pdf/Thomas-Chiang-CV-DS.pdf"}},
			metaData:{fileName: "Thomas-Chiang-CV-DS.pdf"}
		}, {embedMode: "IN_LINE"});
	});
</script>

<!-- ### Certificates -->

<!---
change dir in certificates
-->

<!-- {% include image-gallery-rect.html folder="certificates/datacamp" %}

{% include image-gallery-rect.html folder="certificates/kaggle" %} -->

<!-- ### Published Papers

- Author:
- Co-Author:
	- [Hayes, J J C, E Kerins, J S Morgan et al. (2021)](https://arxiv.org/pdf/2103.12139.pdf) “TransitFit: an exoplanet transit fitting package
for multi-telescope datasets and its application to WASP-127 b, WASP-91 b, and WASP-126 b”, *arXiv*, 1–14. -->

### Introduction

Hi there, My name is Thomas Chiang (or you can call me by my original name, 'Issue' Chiang) and I am a Data Science graduate from the University of Bath with a solid background in chemical engineering.  You might wonder, and all HRs wonder, why am I doing data science for my master's when I did my bachelor's degree in chemical engineering? What made you switch from being a "glorious plumber" to "machine learning code chad?" Well... It's quite a long story, but hear me out.

I have just completed my 4-year bachelor's degree in Chemical Engineering at Heriot-Watt University. The experience at the university was fun and I made a lot of friends, nationally and internationally. I learned so much technical knowledge of chemical engineering I feel like I can just begin my career in it. Edinburgh was cold, but I love the weather (don't judge, I just like the cold). But it was March 2020 and due to the brewing pandemic, I couldn't stay in my accommodation for much longer and was constantly warned by my parents that "I need to leave now". While watching everyone leaving the university accommodation one by one until I am completely alone, I used some of my time to look for job roles related to my degree, as any graduate would do. To my surprise at the time, many of the chemical engineering industries weren't recruiting, and instead, I was bombarded by job roles related to software engineer and data science. That was my first encounter with data science. I started to research this subject and realised the data industry was blooming. The more I read and research about this industry, the more interest I developed in data science. At the time, I knew that if I invest more time in data science roles, not only I would understand more about AI technology, but I can also be part of the change. Therefore, in October 2020, I decided to switch paths from chemical engineering to data science to expand my knowledge about it.

### Experiences and skill sets

I obtained my undergraduate degree in chemical engineering at Heriot-Watt University (1st Class honours) and became very experienced in technical aspects of engineering, the business aspect of different projects and utilising software such as Aspen Plus and Aspen HYSYS. I was involved in numerous design projects during my academic years at Heriot-Watt University where I propose solutions to solve real-world problems. An example of this would be to convert carbon dioxide to sustainable jet fuel, as Boeing and British Airway wish to create an alternative method of obtaining jet fuel to combat climate change and the increase in prices of carbon taxes. As a group, we were able to propose this idea to Exxonmobil and they were pleased with the presentation.

I believe I would be a very suitable candidate for a data scientist related role. I wrote programming languages such as python and SQL, as well as python programming libraries such as NumPy, pandas, TensorFlow, Scikit-Learn, sqlite3 etc. During the academic year in Bath, I applied and implemented data preprocesses and prepare the data for training and testing purposes. Analytical techniques were also used in the context of defining objectives and interpreting results for the general public. In the machine learning part of the course, I was able to display knowledge of algorithmic machine learning approaches and demonstrate an understanding of the application to specific real-world problems and produce a practical implementation of the model to the problem. I also evaluate critically the advantages and the limitations of the approaches made and further improve the model. 

In addition, I have developed a wide range of transferable skills during my gap year that would make me an invaluable addition to any company. I did an internship in Banking in South Africa at the end of 2020, where I obtained knowledge of general operations done in commercial banking such as loans and syndication loans, as well as completed FAIS Regulatory Examination 5 where I understood and applied the legislation knowledge into my work. I have also demonstrated teamwork skills by supporting other colleagues in the bank and teaching them about the legislation. Out of interest, I also did another internship at the Aircraft Career Development Association in Taiwan, where I experienced the technical and physical aspects of aircraft maintenance. During the internship, I was able to discuss innovative ideas with my colleagues about different business models and the method of maintaining a business, which was essential not only in the aircraft industry but also applies to any form of business. 

<!-- Before I started academic study I used to run a game server for *Ultima Online*, primarily written in <span style="font-family:monospace;">C++</span>. As such, the server was highly customisable, and with the source code from another server I painfully merged two incompatible SVN’s together to create one. Just by comparison alone I was able to get used to the logic and constructs that formed the language. Browsing through the many lines of code and merging was a very slow process, but it allowed me understand slightly different methods of doing things. It was almost like learning through reverse engineering. I like to follow the concept of learning by doing. Of note, I took interest in the AI constructs and pathing algorithms adopted, to try and create realistic npcs to fill the world.

My undergraduate studies primarily focused on *Mathematics* in its purest form, to which I supplemented this by taking extra modules from *Physics* to practice application. In my last year I took a formal <span style="font-family:monospace;">C++</span> course to take what I had learned from my game server days and apply it to what I had learned through undergraduate study. Suddenly very nasty equations were efficiently solved! I think data analysis in this aspect fascinates me. My dissertation project involved the inter-facial perturbations of a spherical bubble, to which an approximate formula was derived. As with most fluid dynamics problems, the solution requires numerical analysis. I was able to apply my previous knowledge and reverse engineering from before using the many libraries and pseudo code available.

My most recent foray into applying programming language is in the field of *Astrophysics*. This required extreme care in the handling and use of very large data sets, extracting the precise information needed and filtering such data through a pipeline. Specifically, I worked in exoplanet research, fitting [light curves](https://youtu.be/vLh9KWns9gE){:.lightbox} (time-series data) to improve on the data sets already proven to exist. The hope is that with a greater sensitivity, more interesting and varied data will provide sightings of [exomoons](https://youtu.be/3Ma1xLz1Asw){:.lightbox}. In this field I have written a a self-contained *python* EDA pipeline called [firefly](https://github.com/sourestdeeds/firefly), which uses a transit fitting program called [TransitFit](https://github.com/joshjchayes/TransitFit). Its capable of fitting [TESS](https://youtu.be/Q4KjvPIbgMI){:.lightbox} lightcurves with a [nested sampling](https://github.com/joshspeagle/dynesty) routine, using a *Bayesian* machine learning approach. In the future I hope to expand the functionality by allowing simultaneous fitting of multiple space based and ground based telescopes. -->

<!-- ### Instagram -->

<!-- {% include instagram.html username="ihsiu_chiang" %} -->
