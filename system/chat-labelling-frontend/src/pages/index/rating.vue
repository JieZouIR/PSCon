<template>
  <div style="width:800px; margin: 100px auto">
    <h1 v-if="role === 'cus' && recommend_info.length !== 0" style="text-align: center">Product Relevance Check</h1>
    <p v-if="role === 'cus' && recommend_info.length !== 0" style="text-align: center; font-size: 14px; margin-bottom: 20px;">
      Please evaluate whether the products recommended by the system are what you want (i.e., relevant).
    </p>
    <h1  v-if="role === 'sys'" style="text-align:center">Rate This Conversation</h1>
    <Divider />
    <!-- 显示搜索结果的部分 -->
<!--    <Button type="primary" @click="fetchSearchedMessage">显示</Button>-->
<!--    <div v-if="searchedMessage">-->
<!--      <h3>Search Result:</h3>-->
<!--      <p>{{ searchedMessage }}</p>-->
<!--    </div>-->
    <div v-for="(item, index) in recommend_info" :key="index" v-if="role === 'cus'">
      <h3>{{ item.name }}</h3>
      <RadioGroup v-model="selectedOptions[index]">
        <Radio :label="1">Relevant</Radio>
        <Radio :label="0">Irrelevant</Radio>
      </RadioGroup>
      <br><br>
    </div>
    <div v-if="role === 'cus'">
      <h1 style="text-align:center">Rate This Conversation</h1>
      <h3>Have you obtained your target products successfully?</h3>
      <RadioGroup v-model="cus_rate.anotheranswer">
        <Radio label="yes">Yes</Radio>
        <Radio label="no">No</Radio>
      </RadioGroup>
      <br><br>
      <h3 slot="label">What do you think of your partner's performance?</h3>
      <Rate allow-half v-model="cus_rate.rate" />
      <br><br>
    </div>
    <!--    <button @click="saveResults">Save Results</button>-->
    <Form :model="formItem" ref="form" label-position="top" :rules="rules" :show-message="false">
<!--      <FormItem prop="goalAchieve" v-if="role==='cus'">-->
<!--        <h3 slot="label">Did you achieve your goal?</h3>-->
<!--        <RadioGroup v-model="formItem.goalAchieve">-->
<!--          <Radio label=1>related</Radio>-->
<!--          <Radio label=0>Unrelated</Radio>-->
<!--        </RadioGroup>-->
<!--      </FormItem>-->
      <FormItem prop="goalUnderstand" v-if="role==='sys'">
        <h3 slot="label">Do you think you understand user's goal?</h3>
        <RadioGroup v-model="formItem.goalUnderstand">
          <Radio label=1>Yes</Radio>
          <Radio label=0>No</Radio>
        </RadioGroup>
      </FormItem>
<!--      <FormItem prop="conversationSearch" v-if="role==='cus'">-->
<!--        <h3 slot="label">Would you like to conduct search through conversations in this way?</h3>-->
<!--        <RadioGroup v-model="formItem.conversationSearch">-->
<!--          <Radio label=1>related</Radio>-->
<!--          <Radio label=0>Unrelated</Radio>-->
<!--        </RadioGroup>-->
<!--      </FormItem>-->
      <FormItem prop="easierWay" v-if="role==='sys'">
        <h3 slot="label">Which one is easier for user to achieve his/her goal?</h3>
        <RadioGroup v-model="formItem.easierWay">
          <Radio label=1>Search</Radio>
          <Radio label=0>Conversation</Radio>
        </RadioGroup>
      </FormItem>
      <FormItem prop="rate" v-if="role==='sys'">
        <h3 slot="label">What do you think of your partner's performance?</h3>
        <Rate allow-half v-model="formItem.rate" />
      </FormItem>
      <FormItem v-if="role==='cus'">
        <Button type="primary" @click="restart">Submit</Button>
        <!-- <Button @click="logout">Submit and Logout</Button> -->
        <!-- <Button @click="close">Submit and Close</Button> -->
      </FormItem>
      <FormItem v-if="role==='sys'">
        <Button type="primary" @click="restartSys">Submit</Button>
        <!-- <Button @click="logout">Submit and Logout</Button> -->
        <!-- <Button @click="close">Submit and Close</Button> -->
      </FormItem>
    </Form>
  </div>
</template>
<script>
export default {
  props: ['role', 'conversationId', 'userId', 'recommend_info'],
  data () {
    return {
      extractedTextList: [],
      cus_rate: {
        anotheranswer: null,
        rate: 0
      },
      searchedMessage: null,
      // recommend_info: [],
      selectedOptions: [],
      formItem: {
        goalAchieve: 0,
        conversationSearch: 0,
        rate: 0,
        goalUnderstand: 0,
        easierWay: 0
      },
      rules: {
        goalAchieve: [ { trigger: 'change' } ],
        conversationSearch: [ { message: 'Please select ', trigger: 'change' } ],
        rate: [ {trigger: 'change',
          validator: (rule, value, callback) => {
            if (!value) {
              return callback(new Error('Please rate your partner!'))
            } else callback()
          }} ],
        goalUnderstand: [ { trigger: 'change' } ],
        easierWay: [ { trigger: 'change' } ]
      }
    }
  },
  methods: {
    restart () {
      // console.log(this.recommend_info.length)
      if (this.selectedOptions.length === this.recommend_info.length && this.cus_rate.anotheranswer != null) {
        this.$refs.form.validate((valid) => {
          if (!valid) {
            this.$Message.error('Please fill in the form!')
            return
          }
          this.$http.post('/api/saveRating', {
            recommend_info: this.recommend_info,
            selectedOptions: this.selectedOptions,
            role: this.role,
            userId: this.userId,
            conversationId: this.conversationId
          }).then((response) => {
            // this.$emit('completed')
          }).catch((e) => {
            this.submitting = false
            this.$Message.error('Something wrong when sending data!')
          })
          this.$http.post('/api/saveRatingSys', {...this.cus_rate, role: this.role, conversationId: this.conversationId}).then((response) => {
            this.$emit('completed')
          }).catch((e) => {
            this.submitting = false
            this.$Message.error('Something wrong when sending data!')
          })
        })
      } else {
        this.$Message.error('Please fill in the form!')
      }
    },
    restartSys () {
      this.$refs.form.validate((valid) => {
        if (!valid) {
          this.$Message.error('Please fill in the form!')
          return
        }
        this.$http.post('/api/saveRatingSys', {...this.formItem, role: this.role, conversationId: this.conversationId}).then((response) => {
          this.$emit('completed')
        }).catch((e) => {
          this.submitting = false
          this.$Message.error('Something wrong when sending data!')
        })
      })
    },
    logout () {
      this.$refs.form.validate((valid) => {
        if (!valid) {
          this.$Message.error('Please fill in the form!')
          return
        }
        this.$http.post('/api/saveRating', {...this.formItem, role: this.role, userId: this.userId, conversationId: this.conversationId}).then((response) => {
          window.location = '/logout'
        }).catch((e) => {
          this.submitting = false
          this.$Message.error('Something wrong when sending data!')
        })
      })
    }
  }
}
</script>
