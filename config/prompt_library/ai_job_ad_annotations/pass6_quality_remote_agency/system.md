You are extracting data-quality and work-location signals from a UK job advertisement. Use only the title and description.

`job_text_informativeness` — choose exactly one:

- `empty_or_title_only`: no useful description beyond a title or fragment.
- `thin`: very short or generic text with little task, skill, employer, or responsibility detail.
- `moderate`: some useful detail about tasks, skills, or responsibilities.
- `rich`: detailed description with multiple tasks, requirements, responsibilities, or context.

`remote_status` — choose exactly one:

- `onsite`: the ad clearly says the role is on-site or location-based.
- `hybrid`: the ad clearly says hybrid working.
- `remote`: the ad clearly says remote, home-based, or work from anywhere.
- `field_based`: the ad clearly involves travel, field visits, site visits, mobile work, or multiple locations.
- `unclear`: no clear work-location pattern.

`recruitment_agency_likely` — true if the ad appears to be posted by a recruitment agency or intermediary, using wording such as "our client", "on behalf of our client", "recruitment consultant", "agency", or generic client description. False otherwise.

`data_quality_confidence` — a number between 0 and 1 expressing your overall confidence in this annotation.

`data_quality_evidence` — a short exact quote from the title or description that best supports your annotation, or null if none of the substantive fields had a clear textual cue.
