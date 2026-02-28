import streamlit as st
def render_intro_sections():
    # ---------------------------------------------------
    # What We Do
    # ---------------------------------------------------
    st.markdown("""
    <div class="section-heading">What We Do?</div>
    <div class="section-underline"></div>

    <p class="section-text">
    This system performs <span class="highlight">fraud detection</span> using a 
    <span class="highlight">multilayered analytical framework</span> rather than relying solely on textual signals. 
    Each layer operates <span class="highlight">independently</span> across linguistic, behavioral, and temporal dimensions. 
    The outputs from all layers are <span class="highlight">aggregated</span> into a unified fraud score that determines 
    whether a review should be <span class="highlight">flagged</span> or <span class="highlight">not flagged</span>.
    </p>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # How We Do It
    # ---------------------------------------------------
    st.markdown("""
    <div class="section-heading">How We Do It?</div>
    <div class="section-underline"></div>

    <ul class="section-text custom-list">

    <li>
    <span class="highlight-core">Textual Analysis :</span> 
    Evaluates <span class="highlight-core">structural indicators</span>, 
    <span class="highlight-core">sentiment extremity</span>, 
    <span class="highlight-core">repetition patterns</span>, 
    <span class="highlight-core">promotional keyword density</span>, 
    and absence of <span class="highlight-core">product-specific details</span>.  Review text is transformed using 
    <span class="highlight-tech">TF-IDF vectorization</span> to extract weighted term features, 
    and classified using a 
    <span class="highlight-tech">Logistic Regression model</span> 
    to estimate manipulation probability. Example: Reviews dominated by phrases like 
    <span class="highlight-core">"must buy"</span> or 
    <span class="highlight-core">"life changing experience"</span> 
    without product features are flagged based on model confidence threshold.
    </li>

    <li>
    <span class="highlight-core">Behavioral Analysis :</span> 
    Computes reviewer-level features including 
    <span class="highlight-core">burst activity</span>, 
    <span class="highlight-core">rating deviation</span>, 
    <span class="highlight-core">review frequency anomalies</span>, 
    and <span class="highlight-core">account maturity risk</span>.  A <span class="highlight-tech">feature-based anomaly scoring mechanism</span> 
    normalizes and aggregates behavioral indicators into a composite fraud risk score. Example: A newly created account posting multiple 5-star reviews within minutes 
    is flagged due to abnormal behavioral feature aggregation.
    </li>

    <li>
    <span class="highlight-core">Temporal Analysis :</span> 
    Identifies 
    <span class="highlight-core">review spikes</span>, 
    <span class="highlight-core">clustered submissions</span>, 
    and <span class="highlight-core">abnormal time gaps</span>.  We apply 
    <span class="highlight-tech">time-series based fraud detection</span> 
    using spike detection and temporal density modeling to detect coordinated campaigns. Example: A product receiving 50 reviews within one hour after weeks of inactivity 
    is flagged due to statistically significant temporal deviation.
    </li>

    </ul>
    """, unsafe_allow_html=True)
