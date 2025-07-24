import json
import csv
from typing import List, Dict, Any, Optional
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Class để load và xử lý các loại dataset khác nhau cho Multi-Hop QA
    Hỗ trợ các format: JSON, CSV, TSV, và các dataset chuẩn như HotpotQA
    """
    
    def __init__(self):
        self.data = None
        self.questions = []
        self.contexts = []
        self.answers = []
        
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset từ file JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Đã load {len(data)} mẫu từ {file_path}")
            return data
        except Exception as e:
            logger.error(f"Lỗi khi load file JSON: {e}")
            return []
    
    def load_csv(self, file_path: str, delimiter: str = ',') -> List[Dict[str, Any]]:
        """Load dataset từ file CSV/TSV"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    data.append(dict(row))
            logger.info(f"Đã load {len(data)} mẫu từ {file_path}")
            return data
        except Exception as e:
            logger.error(f"Lỗi khi load file CSV: {e}")
            return []
    
    def load_hotpotqa_format(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset theo format HotpotQA
        Format: {
            "_id": "unique_id",
            "question": "What is the question?",
            "answer": "answer text", 
            "context": [
                ["Title1", ["Sentence 1.", "Sentence 2."]],
                ["Title2", ["Sentence 3.", "Sentence 4."]]
            ],
            "supporting_facts": [["Title1", 0], ["Title2", 1]],
            "type": "bridge|comparison",
            "level": "easy|medium|hard"
        }
        """
        data = self.load_json(file_path)
        processed_data = []
        
        for item in data:
            if 'question' in item and 'context' in item:
                # Xử lý context từ format HotpotQA
                contexts = []
                for title, sentences in item['context']:
                    # Ghép các câu thành đoạn văn
                    paragraph = ' '.join(sentences)
                    contexts.append({
                        'title': title,
                        'text': paragraph
                    })
                
                processed_item = {
                    'id': item.get('_id', f"hotpot_{len(processed_data)}"),
                    'question': item['question'],
                    'answer': item.get('answer', ''),
                    'contexts': contexts,
                    'type': item.get('type', 'bridge'),
                    'level': item.get('level', 'medium'),
                    'dataset': 'hotpotqa',
                    # Preserve original fields including supporting_facts
                    'supporting_facts': item.get('supporting_facts', []),
                    'original': item  # Keep full original data for reference
                }
                processed_data.append(processed_item)
        
        logger.info(f"Đã xử lý {len(processed_data)} mẫu HotpotQA")
        return processed_data
    
    def load_musique_format(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset theo format MusiQue
        Format: {
            "id": "unique_id",
            "question": "What is the question?",
            "answers": ["answer1", "answer2"],
            "paragraphs": [
                {
                    "title": "Title",
                    "paragraph_text": "paragraph content...",
                    "is_supporting": true/false
                }
            ],
            "question_decomposition": [...],
            "answerable": true/false
        }
        """
        processed_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        if 'question' in item and 'paragraphs' in item:
                            # Xử lý paragraphs từ format MusiQue
                            contexts = []
                            for para in item['paragraphs']:
                                contexts.append({
                                    'title': para.get('title', 'Untitled'),
                                    'text': para.get('paragraph_text', ''),
                                    'is_supporting': para.get('is_supporting', False)
                                })
                            
                            # Lấy answer đầu tiên nếu có nhiều answers
                            answer = ""
                            if 'answers' in item and item['answers']:
                                answer = item['answers'][0]
                            elif 'answer' in item:
                                answer = item['answer']
                            
                            processed_item = {
                                'id': item.get('id', f"musique_{len(processed_data)}"),
                                'question': item['question'],
                                'answer': answer,
                                'contexts': contexts,
                                'type': 'multi-hop',
                                'level': 'hard' if not item.get('answerable', True) else 'medium',
                                'dataset': 'musique',
                                'answerable': item.get('answerable', True),
                                'decomposition': item.get('question_decomposition', [])
                            }
                            processed_data.append(processed_item)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Lỗi JSON tại dòng {line_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Lỗi khi load file MusiQue: {e}")
            return []
        
        logger.info(f"Đã xử lý {len(processed_data)} mẫu MusiQue")
        return processed_data
    
    def load_custom_format(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset theo format tự định nghĩa
        Format: {
            "question": "câu hỏi",
            "answer": "câu trả lời",
            "passages": [{"title": "...", "text": "..."}]
        }
        """
        data = self.load_json(file_path)
        processed_data = []
        
        for item in data:
            if 'question' in item and 'passages' in item:
                processed_item = {
                    'question': item['question'],
                    'answer': item.get('answer', ''),
                    'contexts': item['passages'],
                    'type': item.get('type', 'multi-hop'),
                    'level': item.get('level', 'medium')
                }
                processed_data.append(processed_item)
        
        logger.info(f"Đã xử lý {len(processed_data)} mẫu custom format")
        return processed_data
    
    def create_sample_dataset(self, output_path: str = None) -> List[Dict[str, Any]]:
        """Tạo sample dataset English để test (HotpotQA/MusiQue style)"""
        sample_data = [
            {
                "id": "sample_001",
                "question": "What is the capital of the country where the Eiffel Tower is located?",
                "answer": "Paris",
                "contexts": [
                    {"title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought iron lattice tower located in Paris, France. It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair."},
                    {"title": "France", "text": "France is a country in Western Europe. Its capital and largest city is Paris, which is also the country's most populous city."},
                    {"title": "Paris", "text": "Paris is the capital and most populous city of France. It is located in the north-central part of the country."}
                ],
                "type": "bridge",
                "level": "medium",
                "dataset": "sample"
            },
            {
                "id": "sample_002", 
                "question": "Who was the president of the United States when Alaska became a state?",
                "answer": "Dwight D. Eisenhower",
                "contexts": [
                    {"title": "Alaska Statehood", "text": "Alaska was admitted as the 49th state of the United States on January 3, 1959."},
                    {"title": "Dwight D. Eisenhower", "text": "Dwight David Eisenhower was the 34th president of the United States, serving from 1953 to 1961."},
                    {"title": "U.S. States", "text": "The United States consists of 50 states, with Alaska and Hawaii being the last two admitted."}
                ],
                "type": "bridge",
                "level": "hard",
                "dataset": "sample"
            },
            {
                "id": "sample_003",
                "question": "In what year was the university founded that is located in the same city as Harvard Medical School?",
                "answer": "1636",
                "contexts": [
                    {"title": "Harvard Medical School", "text": "Harvard Medical School is the medical school of Harvard University. It is located in Boston, Massachusetts."},
                    {"title": "Harvard University", "text": "Harvard University is a private Ivy League research university in Cambridge, Massachusetts. It was founded in 1636."},
                    {"title": "Boston", "text": "Boston is the capital and most populous city of Massachusetts. It is home to several prestigious universities."},
                    {"title": "Cambridge, Massachusetts", "text": "Cambridge is a city in Massachusetts, directly across the Charles River from Boston. It is home to Harvard University and MIT."}
                ],
                "type": "bridge",
                "level": "hard",
                "dataset": "sample"
            },
            {
                "id": "sample_004",
                "question": "What is the largest city in the state where Mount Rushmore is located?",
                "answer": "Sioux Falls",
                "contexts": [
                    {"title": "Mount Rushmore", "text": "Mount Rushmore National Memorial is a sculpture carved into the granite face of Mount Rushmore in South Dakota."},
                    {"title": "South Dakota", "text": "South Dakota is a state in the Midwestern region of the United States. Its largest city is Sioux Falls."},
                    {"title": "Sioux Falls", "text": "Sioux Falls is the most populous city in South Dakota and the 47th-most populous city in the United States."}
                ],
                "type": "bridge", 
                "level": "medium",
                "dataset": "sample"
            },
            {
                "id": "sample_005",
                "question": "Which film won the Academy Award for Best Picture in the same year that the Berlin Wall fell?",
                "answer": "Driving Miss Daisy",
                "contexts": [
                    {"title": "Fall of the Berlin Wall", "text": "The Berlin Wall fell on November 9, 1989, marking the beginning of the end of the Cold War."},
                    {"title": "62nd Academy Awards", "text": "The 62nd Academy Awards ceremony was held on March 26, 1990, honoring films released in 1989. Driving Miss Daisy won Best Picture."},
                    {"title": "Driving Miss Daisy", "text": "Driving Miss Daisy is a 1989 American comedy-drama film directed by Bruce Beresford and written by Alfred Uhry."}
                ],
                "type": "bridge",
                "level": "hard", 
                "dataset": "sample"
            }
        ]
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã tạo English sample dataset tại {output_path}")
        
        return sample_data
    
    def get_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Thống kê dataset"""
        if not data:
            return {}
        
        stats = {
            'total_samples': len(data),
            'question_types': {},
            'difficulty_levels': {},
            'avg_contexts_per_question': 0,
            'avg_question_length': 0,
            'avg_context_length': 0
        }
        
        total_contexts = 0
        total_question_length = 0
        total_context_length = 0
        
        for item in data:
            # Thống kê loại câu hỏi
            q_type = item.get('type', 'unknown')
            stats['question_types'][q_type] = stats['question_types'].get(q_type, 0) + 1
            
            # Thống kê độ khó
            level = item.get('level', 'unknown')
            stats['difficulty_levels'][level] = stats['difficulty_levels'].get(level, 0) + 1
            
            # Thống kê độ dài
            if 'contexts' in item:
                contexts_count = len(item['contexts'])
                total_contexts += contexts_count
                
                for ctx in item['contexts']:
                    total_context_length += len(ctx.get('text', '').split())
            
            if 'question' in item:
                total_question_length += len(item['question'].split())
        
        # Tính trung bình
        stats['avg_contexts_per_question'] = total_contexts / len(data)
        stats['avg_question_length'] = total_question_length / len(data)
        stats['avg_context_length'] = total_context_length / total_contexts if total_contexts > 0 else 0
        
        return stats
    
    def print_statistics(self, data: List[Dict[str, Any]]):
        """In thống kê dataset"""
        stats = self.get_statistics(data)
        
        print("\n" + "="*50)
        print("THỐNG KÊ DATASET")
        print("="*50)
        print(f"Tổng số mẫu: {stats['total_samples']}")
        print(f"Số context trung bình mỗi câu hỏi: {stats['avg_contexts_per_question']:.2f}")
        print(f"Độ dài câu hỏi trung bình: {stats['avg_question_length']:.2f} từ")
        print(f"Độ dài context trung bình: {stats['avg_context_length']:.2f} từ")
        
        print("\nLoại câu hỏi:")
        for q_type, count in stats['question_types'].items():
            print(f"  {q_type}: {count} ({count/stats['total_samples']*100:.1f}%)")
        
        print("\nĐộ khó:")
        for level, count in stats['difficulty_levels'].items():
            print(f"  {level}: {count} ({count/stats['total_samples']*100:.1f}%)")
        print("="*50)

# Convenience functions for direct dataset loading
def load_hotpot_data(split='dev', sample_size=None):
    """
    Load HotpotQA data directly
    Args:
        split: 'dev', 'train' for dataset split
        sample_size: limit number of samples (None for all)
    """
    loader = DatasetLoader()
    
    # Map split to filename
    if split == 'dev':
        file_path = "data/hotpotqa/hotpot_dev_distractor_v1.json"
    elif split == 'train':
        file_path = "data/hotpotqa/hotpot_train_v1.1.json"
    else:
        file_path = f"data/hotpotqa/hotpot_{split}_v1.json"
    
    try:
        data = loader.load_hotpotqa_format(file_path)
        
        if sample_size and len(data) > sample_size:
            data = data[:sample_size]
            logger.info(f"Limited to {sample_size} samples")
        
        return data
    except Exception as e:
        logger.error(f"Error loading HotpotQA {split} data: {e}")
        return []

def load_musique_data(sample_size=None):
    """Load MusiQue data directly"""
    loader = DatasetLoader()
    
    try:
        data = loader.load_musique_format("data/musique_sample.jsonl")
        
        if sample_size and len(data) > sample_size:
            data = data[:sample_size]
            logger.info(f"Limited to {sample_size} samples")
        
        return data
    except Exception as e:
        logger.error(f"Error loading MusiQue data: {e}")
        return []

# Ví dụ sử dụng
if __name__ == "__main__":
    loader = DatasetLoader()
    
    # Tạo sample dataset
    sample_data = loader.create_sample_dataset("data/sample_dataset.json")
    
    # In thống kê
    loader.print_statistics(sample_data)
    
    # Test load lại
    loaded_data = loader.load_custom_format("data/sample_dataset.json")
    print(f"\nĐã load lại {len(loaded_data)} mẫu từ file")
