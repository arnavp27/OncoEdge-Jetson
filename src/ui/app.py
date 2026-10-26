"""A sample-first workspace for inspecting the real OncoEdge pipeline."""
import hashlib
import html
import io
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.pipeline.inference_pipeline import OncoEdgePipeline
from src.pipeline.demo_readout import DemoOptions
from src.config import CONFIG


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets'
LOGGER = logging.getLogger(__name__)
MAX_PIXELS = 16_000_000


@st.cache_resource(show_spinner=False)
def load_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return OncoEdgePipeline(device=device)


def load_samples():
    return json.loads((ASSETS / 'samples/catalog.json').read_text())


def read_image(data):
    """Decode and orient an image without altering its model-facing colors."""
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ValueError('Please choose an image smaller than 16 megapixels.')
        if min(source.size) < 32:
            raise ValueError('Please choose an image at least 32 pixels on each side.')
        return ImageOps.exif_transpose(source).convert('RGB')


@st.cache_data(show_spinner=False)
def thumbnail(filename):
    import base64

    image = read_image((ASSETS / 'samples' / filename).read_bytes())
    image.thumbnail((320, 180), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def input_key(image, metadata, source_id, settings=None):
    digest = hashlib.sha256(image.tobytes())
    digest.update(str(image.size).encode())
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    digest.update(source_id.encode())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def select_sample(sample_id):
    st.session_state.selected_sample = sample_id


def render_header():
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    st.html(f'''<div class="masthead">
      <div class="identity"><span class="brand-mark" aria-hidden="true">o<span>e</span></span>
        <div><div class="wordmark">OncoEdge<span> / </span></div>
        <div class="brand-caption">EDGE AI · RESEARCH PROTOTYPE</div></div></div>
      <div class="masthead-links"><span class="runtime-pill"><i></i>Live inference · {device}</span>
        <a href="https://github.com/arnavp27/OncoEdge-Jetson" target="_blank" rel="noopener noreferrer">View project ↗</a></div>
    </div>
    <section class="hero">
      <div><p class="eyebrow">COMPUTER VISION, AT A HUMAN SCALE</p>
        <h1>Oral lesion analysis.<br><em>Built for the edge.</em></h1>
        <p class="hero-copy">A closer look at how detection and biomedical image classification
        work together. Start with a sample, or bring your own photograph.</p></div>
      <div class="pipeline-map" aria-label="Pipeline: YOLO11 locates regions, BioMedCLIP compares image and text, then outputs are reviewed">
        <div class="map-label">ONE IMAGE. TWO MODELS.</div>
        <div class="map-row"><span class="map-number">01</span><div><b>Locate</b><small>YOLO11 · region detection</small></div><span class="map-arrow">↘</span></div>
        <div class="map-row"><span class="map-number">02</span><div><b>Compare</b><small>BioMedCLIP · image + text</small></div><span class="map-arrow">↘</span></div>
        <div class="map-row"><span class="map-number">03</span><div><b>Review</b><small>Labels, evidence &amp; uncertainty</small></div><span class="map-dot"></span></div>
      </div>
    </section>''')


def section_label(number, title, detail=''):
    st.html(f'<div class="section-label"><span>{html.escape(number)}</span>'
            f'<h2>{html.escape(title)}</h2><small>{html.escape(detail)}</small></div>')


def sample_gallery(samples):
    st.caption('Six published examples. Each runs the live pipeline; no results are prefilled.')
    with st.container(key='sample_gallery'):
        columns = st.columns(len(samples), gap='small')
        for column, sample in zip(columns, samples):
            selected = sample['id'] == st.session_state.selected_sample
            with column, st.container(border=True, key='sample_' + sample['id'].replace('-', '_')):
                st.html(f'<div class="sample-photo"><img src="{thumbnail(sample["file"])}" '
                        f'alt="{html.escape(sample["title"])}"></div>')
                st.button(
                    sample['title'], key='pick_' + sample['id'],
                    type='primary' if selected else 'secondary', width='stretch',
                    on_click=select_sample, args=(sample['id'],),
                    help='Selected sample' if selected else 'Use this photograph',
                )


def empty_results():
    st.html('''<div class="empty-result">
      <div class="viewfinder" aria-hidden="true"><span></span><i></i></div>
      <p class="eyebrow">READY WHEN YOU ARE</p>
      <h3>See what the models see.</h3>
      <p>Choose a photograph and run analysis. Detected regions, both model labels,
      and any uncertainty will appear here.</p>
      <div class="empty-steps"><span>01 Choose an image</span><span>02 Run analysis</span><span>03 Inspect the evidence</span></div>
    </div>''')
    with st.expander('What should I expect?'):
        st.write('YOLO locates candidate regions. BioMedCLIP compares each crop with four '
                 'text prompts: OCA (oral cancer), OPMD (potentially malignant disorder), '
                 'Benign and Normal. The models can miss regions or give conflicting labels.')
        st.write('The result shows the classifier label from the region with the strongest '
                 'combined model match. Open a region to compare both models and its scores. '
                 'An image without a usable region has no classification result.')


def acceptance_controls():
    exploratory = not st.toggle('Require model agreement', value=False, key='require_model_agreement',
                                help='Withhold an assessment when labels disagree or scores are ambiguous.')
    if exploratory:
        detector = st.slider('Minimum YOLO confidence', 5, 50, 10, 1,
                             format='%d%%', key='demo_detector_limit')
        classifier = st.slider('Minimum classifier relative score', 25, 100, 35, 1,
                               format='%d%%', key='demo_classifier_limit')
        margin = st.slider('Minimum top-two score gap', 0, 50, 2, 1,
                           format='%d%%', key='demo_margin_limit')
        st.caption('Lower limits include more candidates. The summary uses the strongest '
                   'detector × classifier match; both labels remain available in the results.')
        options = DemoOptions(detector / 100, classifier / 100, margin / 100)
        settings = {'mode': 'exploratory_demo', **asdict(options)}
    else:
        st.caption('Unresolved evidence returns INCONCLUSIVE with no assessment score.')
        options = None
        config = CONFIG['model']
        settings = {
            'mode': 'conservative_review', 'detector_threshold': config.yolo_conf_threshold,
            'min_classifier_score': config.min_classifier_score,
            'min_classifier_margin': config.min_classifier_margin,
        }
    return options, settings


def display_results(results, elapsed=None, backend=None):
    """Explain a real pipeline result without presenting matches as diagnoses."""
    level = results['risk_level']
    descriptions = {
        'INCONCLUSIVE': 'The evidence needs review',
        'HIGH': 'The models flagged concerning features',
        'MEDIUM': 'The models flagged a potentially malignant appearance',
        'LOW': 'The models agreed on non-concerning labels',
    }
    demo = results.get('demo_readout')
    issues = results.get('assessment_issues', [])
    if demo is not None:
        st.html('<div class="result-eyebrow"><span class="result-dot"></span>Model results</div>')
        if demo['class'] is not None:
            st.subheader(f'Model label: {demo["class"]}')
            st.caption(f'Region {demo["primary_region"]} · Strongest combined match')
        else:
            st.subheader('No label available')
            st.info('No usable region passed the analysis settings. Try another photograph '
                    'or adjust the settings.')
        with st.expander('Model checks'):
            st.write(f'**Assessment status:** {level}')
            for issue in issues:
                st.write('- ' + issue)
            if not issues:
                st.write('The detected regions passed the model agreement checks.')
            st.caption('The classifier summary can be shown even when these checks are unresolved.')
            if demo['concern_index'] is not None:
                st.write(f'**Concern index:** {demo["concern_index"]:.1f}/10')
                st.caption('Detector-weighted OCA + OPMD evidence across included regions; '
                           'uncalibrated and without patient-context weighting.')
    else:
        st.html(f'<div class="result-eyebrow"><span class="result-dot {html.escape(level.lower())}"></span>'
                f'{html.escape(descriptions.get(level, "Research assessment"))}</div>')
        st.subheader(f'Assessment: {level}')
        if level == 'HIGH':
            st.error(results['recommendation'], icon=':material/info:')
        elif level in ('INCONCLUSIVE', 'MEDIUM'):
            st.warning(results['recommendation'], icon=':material/info:')
        else:
            st.info(results['recommendation'], icon=':material/info:')
        if issues:
            with st.expander('Why this result?', expanded=len(issues) <= 2):
                for issue in issues:
                    st.write('- ' + issue)

    with st.container(key='result_metrics'):
        columns = st.columns(3)
        if demo is not None:
            score = demo['classifier_score']
            columns[0].metric('Text match', 'Not assessed' if score is None else f'{score:.1%}',
                              help='Relative BioMedCLIP score across four text labels, not a disease probability.')
        else:
            score = results['risk_score']
            columns[0].metric('Concern index', 'Not assessed' if score is None else f'{score:.1f}/10')
        columns[1].metric('Regions detected', len(results['detections']))
        columns[2].metric('Analysis time', f'{elapsed:.1f}s' if elapsed is not None else '—')
    st.caption('Scores are relative model outputs, not disease probabilities.'
               + (' Time includes model loading on the first run.' if elapsed is not None else ''))

    if results['detections']:
        st.markdown('#### Detected regions')
        with st.container(key='region_preview'):
            st.image(results['visualization'], width='stretch',
                     caption='Model-generated overlays. A ? marks a region that needs review.')
        st.markdown('#### Inspect the evidence')
        for i, det in enumerate(results['detections']):
            needs_review = (det['model_disagreement'] or det['classification_uncertain']
                            or det.get('detection_uncertain', False))
            title = f'Region {i+1} · {det["class"]}'
            with st.expander(title):
                st.write(f'**Detector label:** {det["yolo_class"]}')
                st.write(f'**Classifier label:** {det["class"]}')
                st.write(f'**YOLO confidence:** {det["yolo_conf"]:.1%} · '
                         f'**Fused match:** {det["fusion_score"]:.1%}')
                if demo is not None:
                    st.caption('Included in the summary.' if i + 1 in demo['accepted_regions']
                               else 'Below the selected limits; excluded from the summary.')
                if needs_review:
                    reasons = []
                    if det['model_disagreement']:
                        reasons.append('The detector and classifier give different labels.')
                    if det['classification_uncertain']:
                        reasons.append('The classifier scores are ambiguous.')
                    if det.get('detection_uncertain'):
                        reasons.append('Detector confidence is below the agreement-check limit.')
                    st.warning(' '.join(reasons))
                st.write('**Classifier relative scores (not disease probabilities):**')
                for label, value in det['class_scores'].items():
                    st.progress(min(1.0, max(0.0, value)), text=f'{label} · {value:.1%}')
    else:
        st.info('No regions were available for classification. Disease has not been ruled out.')
    if backend:
        st.caption(f'Live inference · {backend.upper()} · trained YOLO checkpoint + BioMedCLIP')


def report_json(analysis):
    result = analysis['result']
    return json.dumps({
        'project': 'OncoEdge', 'research_only': True,
        'image': analysis['image_name'], 'demo_context': analysis['metadata'],
        'backend': analysis['backend'], 'elapsed_seconds': round(analysis['elapsed'], 3),
        'risk_level': result['risk_level'], 'risk_score': result['risk_score'],
        'recommendation': result['recommendation'],
        'assessment_issues': result.get('assessment_issues', []),
        'inference_settings': analysis.get('settings', {'mode': 'conservative_review'}),
        'demo_readout': result.get('demo_readout'),
        'detections': [{k: v for k, v in det.items() if k != 'mask'}
                       for det in result['detections']],
    }, indent=2)


def main():
    st.set_page_config(page_title='OncoEdge · Oral image analysis', page_icon='◉',
                       layout='wide', initial_sidebar_state='collapsed')
    st.html('<style>' + (ASSETS / 'styles.css').read_text() + '</style>')
    samples = load_samples()
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = samples[0]['id']
    if st.session_state.selected_sample not in {s['id'] for s in samples}:
        st.session_state.selected_sample = samples[0]['id']
    render_header()

    section_label('01', 'Choose an image', 'SAMPLE LIBRARY / UPLOAD')
    with st.container(key='source_selector'):
        source = st.radio('Image source', ['Samples', 'Upload'], horizontal=True,
                          key='image_source', label_visibility='collapsed')
    image, selected, image_name, source_id = None, None, '', ''
    if source == 'Samples':
        sample_gallery(samples)
        selected = next(s for s in samples if s['id'] == st.session_state.selected_sample)
        try:
            image = read_image((ASSETS / 'samples' / selected['file']).read_bytes())
            image_name, source_id = selected['title'], selected['id']
        except (OSError, ValueError, UnidentifiedImageError):
            st.error('This sample could not be opened. Please choose another photograph.')
    else:
        uploaded = st.file_uploader('Add an oral photograph', type=['jpg', 'jpeg', 'png', 'webp'],
                                    help='JPEG, PNG or WebP · up to 20 MB and 16 megapixels',
                                    key='image_upload')
        if uploaded is not None:
            try:
                image = read_image(uploaded.getvalue())
                image_name, source_id = uploaded.name, 'upload:' + uploaded.name
            except ValueError as error:
                st.error(str(error))
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                st.error('This file could not be read as an image. Try another JPEG, PNG or WebP.')

    with st.container(key='workbench'):
        left, right = st.columns([1, 1.08], gap='large')
        with left:
            section_label('02', 'Your photograph')
            with st.container(border=True, key='input_panel'):
                if image is not None:
                    with st.container(key='input_preview'):
                        st.image(image, width='stretch')
                    st.html(f'<div class="photo-name">{html.escape(image_name)}</div>')
                    st.caption(f'{image.width:,} × {image.height:,} px · '
                               + ('Published sample' if selected else 'Uploaded photograph'))
                else:
                    st.html('<div class="upload-empty">Your photograph will appear here.<br>'
                            '<small>Choose a file above to get started.</small></div>')

                with st.expander('Analysis settings', expanded=False):
                    demo_options, settings = acceptance_controls()
                    st.markdown('**Patient context**')
                    st.caption('Optional inputs for the rule-based assessment. Defaults are '
                               'not the photographed subjects’ medical histories and do not '
                               'affect the classifier label or text match.')
                    context_columns = st.columns(2)
                    age = context_columns[0].number_input('Age', min_value=1, max_value=120,
                                                          value=45, key='demo_age')
                    tobacco_years = context_columns[1].number_input(
                        'Years of tobacco use', min_value=0, max_value=120, value=0,
                        key='demo_tobacco')
                    duration = st.selectbox('Lesion duration', ['< 2 weeks', '2-6 weeks', '> 6 weeks'],
                                             key='demo_duration')
                metadata = {'age': age, 'tobacco_years': tobacco_years, 'lesion_duration': duration}
                valid_context = tobacco_years <= age
                if not valid_context:
                    st.warning('Years of tobacco use cannot exceed age.')
                fingerprint = input_key(image, metadata, source_id, settings) if image is not None else None
                analysis = st.session_state.get('analysis')
                stale = analysis is not None and analysis['input_key'] != fingerprint
                if stale:
                    st.session_state.pop('analysis', None)
                    analysis = None
                run = st.button('Run analysis', type='primary', icon=':material/arrow_forward:',
                                width='stretch', key='run_analysis',
                                disabled=image is None or not valid_context)
                st.caption('Runs the actual models. The first analysis takes longer while they load.')
                if run:
                    st.session_state.pop('analysis', None)
                    analysis = None
                    try:
                        started = time.perf_counter()
                        with st.status('Analyzing photograph…', expanded=True) as status:
                            st.write('Loading the detection and classification models…')
                            pipeline = load_pipeline()
                            st.write('Locating regions and comparing biomedical image prompts…')
                            result = pipeline.process_image(np.array(image), metadata,
                                                            demo_options=demo_options)
                            analysis = {
                                'input_key': fingerprint, 'result': result,
                                'elapsed': time.perf_counter() - started,
                                'backend': pipeline.backend, 'image_name': image_name,
                                'metadata': metadata, 'settings': settings,
                            }
                            st.session_state.analysis = analysis
                            status.update(label='Analysis complete', state='complete', expanded=False)
                    except Exception:
                        LOGGER.exception('OncoEdge analysis failed')
                        st.error('Analysis could not finish. Try again or choose another image. '
                                 'Details are available in the local application logs.')
            if selected:
                with st.expander('About this sample'):
                    st.write(selected['description'])
                    st.write(f'**Published reference:** {selected["source_label"]}')
                    st.caption('The publication’s reference is context, not a model prediction '
                               'or an independently verified diagnosis.')
                    st.markdown(f'[{selected["source_credit"]}, figure {selected["figure"]}]'
                                f'({selected["source_url"]}) · [CC BY 4.0]'
                                '(https://creativecommons.org/licenses/by/4.0/)')

        with right:
            section_label('03', 'Analysis', 'LIVE MODEL OUTPUT')
            with st.container(border=True, key='analysis_panel'):
                if analysis is not None:
                    display_results(analysis['result'], analysis['elapsed'], analysis['backend'])
                    st.download_button('Download analysis JSON', report_json(analysis),
                                       file_name='oncoedge-analysis.json', mime='application/json',
                                       icon=':material/download:', width='stretch')
                else:
                    if stale:
                        st.info('Inputs changed. Run analysis again to update the result.')
                    empty_results()

    with st.expander('About OncoEdge'):
        st.write('OncoEdge explores oral image analysis on resource-constrained edge hardware. '
                 'This interface runs the trained YOLO11 detector and zero-shot BioMedCLIP classifier '
                 'using the available local backend. It is an engineering research prototype.')
        st.write('Results show relative matches between image regions and four text labels. '
                 'This is not a medical diagnosis. Evaluation details and known limitations '
                 'are documented in the project repository.')
    st.html('<footer class="page-footer"><span>OncoEdge / research in progress</span>'
            '<span>YOLO11 + BioMedCLIP</span></footer>')
