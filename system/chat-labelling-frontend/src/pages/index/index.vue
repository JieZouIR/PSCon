<template>
  <div>
    <Modal v-model="showInstructions" draggable scrollable title="Instructions">
      <div v-html="instructions_cus" v-if="role === 'cus'"></div>
      <div v-html="instructions_sys" v-if="role === 'sys'"></div>
      <div slot="footer">
        <Button type="primary" @click="showInstructions=false">OK</Button>
      </div>
    </Modal>
    <profile v-show="needProfile" @completed="profileCompleted"/>
    <Card :bordered="false" dis-hover v-show="!finished&&!needProfile">
      <p slot="title" style="text-align:center;overflow:hidden">
        <strong slot="title">
          <Icon type="ios-chatbubbles-outline" :size="20" />
          Current user is {{username}}, chatting with user {{partner}}
        </strong>
      </p>
      <div slot="extra">
        <Button type="text" @click="showInstructions=true">
          <Icon type="ios-information-circle-outline" />
          <span style="font-weight: bold; color: red;">Instructions</span>
        </Button>
      </div>
      <Spin size="large" v-if="loading" fix>
        <Icon type="ios-loading" size="50" class="spin-icon-load"></Icon>
        <div>{{message}}, please don't leave...</div>
      </Spin>
      <div>
        <Row :gutter="8">
          <Col :span="10">
            <chatUI ref="chatUI" :style="{height:(height+35)+'px',overflow:'auto'}" />
          </Col>
          <Col :span="14">
            <Alert type="warning" v-if="!finished">
              <strong>Note:</strong>
              <ul style="padding:0 10px;line-height:150%;font-size:80%">
                <li v-if="disabled">
                  Please wait for your partner's message.
                  <Button
                    style="margin-left:10px"
                    type="primary"
                    size="small"
                    @click="hurryup"
                    :disabled="hurryupLeftSeconds>0"
                  >Hurry up!{{hurryupLeftSeconds>0? ('('+hurryupLeftSeconds+'s)'):''}}</Button>
                </li>
              </ul>
            </Alert>
            <labelling
              :height="(height-20-(role=='cus'? 60:0))+'px'"
              ref="labelling"
              :role="role"
              :currentState="currentState"
              :searchData="searchData"
              :sendData="sendData"
              :sendDisabled="disabled"
              :actions="actions"
              :conversationId="conversationId"
              :finish="finish"
              :searchResultConfig="searchResultConfig"
              :searching="searching"
              :searchPanelLoading="searchPanelLoading"
              @on-filters="searchWithFilters"
              @states-backup="saveStatesBackup"
              @changeBackupForParent="saveChangeBackup"
              @changeisrecommend="changeisrecommend"
              @passselected="passselected"
              @getRecommendInfo="getRecommendInfo"
              :bus="bus"
            />
          </Col>
        </Row>
      </div>
    </Card>
    <rating v-show="finished&&!needProfile" :role="role" :userId="username" :conversationId="conversationId" :recommend_info="recommend_info" @completed="ratingCompleted"/>
  </div>
</template>
<script>
import chatUI from './chatUI'
import profile from './profile'
import labelling from './labelling'
import rating from './rating'
import util from '../../utils/util'
import Vue from 'vue'
import axios from 'axios'
import md5 from 'crypto-js/md5'
const maxHurryupLeftSeconds = 60
export default {
  components: {
    chatUI,
    labelling,
    rating,
    profile
  },
  mounted () {
    this.init()
    window.addEventListener('beforeunload', e => this.beforeunloadFn(e))
  },
  destroyed () {
    window.removeEventListener('beforeunload', e => this.beforeunloadFn(e))
    this.$socket.close()
  },
  data () {
    return {
      raw_info: [],
      add_info: [],
      dialogVisible: false,
      searching: false,
      searchPanelLoading: false,
      recommend_info: [],
      height: window.screen.availHeight - 230,
      loading: false,
      message: '',
      username: '',
      partner: '',
      conversationId: '',
      disabled: false,
      role: '',
      hurryupLeftSeconds: 0,
      actions: [],
      currentState: [],
      finished: false,
      needProfile: false,
      background: '',
      instructions: '',
      showInstructions: false,
      searchResultConfig: {},
      statesBackupList: [],
      bus: new Vue(),
      searchResultsBackup: {},
      selectedResultsBackup: [],
      isrecommend: false,
      instructions_cus: 'Thank you for agreeing to take part of this study, where you will be acting as a normal user role to have a chat. The chat is about seeking for products (normal user role) or recommending products (system role) to buy.<br>Please carefully watch the following operation introduction video:<a href="https://youtu.be/ghOi2Q5JqEI" target="_blank ">Instructions for normal users of chat-labelling platform (1min39s).</a>If the video is not clear, please adjust to the high-definition version on the video viewing page to watch.<br><br>To complete the study, please follow the steps outlined below.<br>1. We would like to know some general information about you, so we will first ask you to complete a survey related to demographic data, i.e., gender, age, and majors.<br>2. You will then have a conversation with the system. The conversation is initiated by you. You need to select an \'intent\' first, and then write a message in the \'response\' to send to the system (by clicking the \'submit\' botton). When selecting an \'intent\', you can see the explanation for each \'intent\' by hovering your mouse over the \'?\' icon beside the \'intent\'. To send multiple messages in one round, be sure to check \'send another message\' before sending a message.<br>3. After sending your messages, please wait for the system to reply or recommend products to you. After you receive the system\'s message, you can repeat step 2 to continue the conversation.<br>4. After you think you are done for the conversation, you can click \'finish the conversation\' to finish the conversation. <br>5. After you click \'finish the conversation\', you will need to <strong>evaluate the relevance of the products</strong> recommended by the system in the conversations. So please make sure you are aware of the relevance of the recommended products with your target product.',
      instructions_sys: 'Thank you for agreeing to take part of this study, where you will be acting as a normal system role to have a chat. The chat is about seeking for products (normal user role) or recommending products (system role) to buy.<br>Please carefully watch the following operation introduction video:<a href="https://youtu.be/EbhRu8xReBY" target="_blank ">Instructions for system users of chat-labelling platform (1min39s).</a>If the video is not clear, please adjust to the high-definition version on the video viewing page to watch. <br><br>To complete the study, please follow the steps outlined below.<br>1. We would like to know some general information about you, so we will first ask you to complete a survey related to demographic data, i.e., gender, age, and majors.<br>2. You will then have a conversation with the user. Please waiting for the user to send the first message. After you get the message from the user, you need to select an \'action\' first. When selecting an \'action\', you can see the explanation for each \'action\' by hovering your mouse over the \'?\' icon beside the \'action\'. You then write a message in the \'response\' (by clicking the \'submit\' botton). To send multiple messages in one round, be sure to check \'send another message\' before sending a message.<br>3. For each round of conversation, you can input a query to search for related results for products. You can also click \'show filters\' to see filters and search for better products based on the filters when necessary. Note that the search engine may be a bit slow, please be patient after you submit your query and wait for the results.<br>4. When you think that there are products meeting the user\'s need, you can select \'Recommend products\' in the \'action\' and select corresponding products from the search result panel (by checking the checkbox right next to the search results of products) to recommend to the user. Write a text message to respond in the \'response\' textbox (e.g., what about the following choices?) and your selection of recommended products will be added automatically after your text message to send to the user. <br>5. When you think you need more information about the user\'s need, you can select \'Ask Clarifying questions\' in the \'action\', select corresponding \'aspects\' from the result panel (by checking the checkbox right next to the \'aspects\'), and ask a clarifying question in the \'response\' textbox about the \'aspects\' (e.g., what brand of cell phone do you like? "Brand" is the aspect you want to ask).<br>6. After the user is done for the conversation and clicks \'finish the conversation\' button, you will need to answer a few questionnaires to evaluate the conversation.'
    }
  },
  methods: {
    changeisrecommend () {
      this.isrecommend = true
      // this.getRecommendInfo()
      console.log(this.recommend_info)
    },
    passselected (selected) {
      // this.raw_info = selected
      console.log('ffffffffffffffffffffffffffffffffffffff')
      console.log(selected)
      this.raw_info = selected.map(item => item.title)
      // for (let i = 0; i < selected.length; i++) {
      //   console.log('item')
      //   console.log(selected[i])
      //   console.log(selected[i].title)
      //   if (!this.add_row.includes(selected[i].title)) {
      //     // 如果不在，将其添加到 this.add_row 中
      //     this.add_row.push(selected[i].title)
      //   }
      // }
      for (let i = 0; i < this.raw_info.length; i++) {
        if (!this.add_info.includes(this.raw_info[i])) {
          this.add_info.push(this.raw_info[i])
        }
      }
      console.log('raw')
      console.log(this.raw_info)
      console.log('add')
      console.log(this.add_info)
      this.recommend_info = this.add_info
    },
    async getRecommendInfo () {
      try {
        // 调用后端API来获取搜索结果消息
        const response = await this.$http.get('/api/getSearchedMessage', {
          params: {
            conversationId: this.conversationId
          }
        })
        if (response.data !== null) {
          this.searchedMessage = response.data // 将搜索结果赋值给数据对象
          const regex = /href="[^"]*qid=(\d+&sr=\d+-\d+)/g // 提取qid和后续内容的正则表达式
          const matches = [...this.searchedMessage.matchAll(regex)] // 匹配所有链接中的qid和后续内容
          const productNames = this.searchedMessage ? this.searchedMessage.match(/>([^<]+)(?=<\/a>)/g).map(item => item.slice(1)) : [] // 提取产品名称
          // console.log(matches)
          console.log(productNames)
          if (productNames.length === 0) {
            this.recommend_info = []
            return
          }
          const recommendInfoMap = {} // 辅助对象用于去重
          this.recommend_info = matches.map((match, index) => {
            const name = productNames[index] // 获取产品名称
            // print(name)
            const id = md5(name).toString() // 提取qid
            if (!recommendInfoMap[name]) {
              recommendInfoMap[name] = true
              return { id, name }
            }
            return null // 已经添加过的id，返回null
          }).filter(item => item !== null)
        }
        // console.log(this.recommend_info)
      } catch (error) {
        console.error('Error fetching searched message:', error)
      }
    },
    beforeunloadFn (e) {
      if (this.finished) return
      // 这个事件只有在鼠标真正和浏览器有了交互，再刷新或者关闭时才会触发
      e = e || window.event
      if (e) {
        e.returnValue = '关闭提示'
      }
      return '关闭提示'
    },
    async init () {
      this.loadUser()
      this.loading = true
      this.message = 'Connecting to server'
      this.initWebSocket()
      // this.message = 'Checking profile'
      await this.checkProfile()
      // this.message = 'Loading instructions'
      await this.loadInstructions()
      // this.message = 'Loading search result config'
      await this.loadSearchResultConfig()
      // this.message = 'Loading actions'
      await this.loadAction()
      // this.message = 'Waiting for commands from server'
      await this.loadRecommend() // 获取推荐信息
    },
    loadRecommend () {
      if (this.role !== 'cus') { // 只有当角色为用户时才查询add_info
        return
      }
      return this.$http.get('/api/loadRecommendInfo', {
        params: {
          conversationId: this.conversationId,
          user: this.username
        }
      }).then((response) => {
        this.add_info = response.data
        this.recommend_info = this.add_info
        console.log(this.recommend_info)
      })
    },
    profileCompleted () {
      this.needProfile = false
    },
    ratingCompleted () {
      window.location = '/endchoice.html'
    },
    loadInstructions () {
      return this.$http.get('/api/instructions').then((response) => {
        this.instructions = response.data
      })
    },
    loadSearchResultConfig () {
      return this.$http.get('/api/loadSearchResultConfig').then((response) => {
        this.searchResultConfig = response.data
      })
    },
    checkProfile () {
      return this.$http.get('/api/checkProfile').then((response) => {
        this.needProfile = response.data
      })
    },
    initWebSocket () {
      let loc = window.location
      this.$connect('ws://' + loc.hostname + ':' + loc.port + '/websocket/' + this.username)
      this.$socket.onopen = this.websocketonopen
      this.$socket.onerror = this.websocketonerror
      this.$socket.onmessage = this.websocketonmessage
      this.$socket.onclose = this.websocketclose
    },
    finish (data, callback) {
      this.loading = true
      this.message = 'Finishing conversation'
      this.websocketsend({type: 'FINISH'})
      callback()
    },
    reset () {
      this.loading = true
      this.message = ''
      this.partner = ''
      this.disabled = false
      this.role = ''
      this.finished = false
      this.$refs.chatUI.reset()
      this.$refs.labelling.reset()
    },
    websocketonopen (e) {
      // console.log('WebSocket连接成功', e)
    },

    websocketonerror (e) {
      // 错误
      this.$Notice.error(
        {
          title: 'Error',
          desc: 'Something wrong on the server!'
        })
    },
    websocketonmessage (e) {
      const dataList = JSON.parse(e.data)
      dataList.map(async data => {
        // 数据接收，同步的执行方式，load数据完之后，才能往下执行
        if (data.messageCommand === 'START' && !this.finished) {
          this.$Notice.info(
            {
              title: 'Notice',
              desc: 'Your partner is online!'
            }
          )
          this.reset()
          let tempData = data.data
          this.partner = tempData.partner
          this.role = tempData.role
          this.conversationId = tempData.conversationId
          // this.userId = tempData.userId
          this.loading = true
          this.message = 'Loading history and Background'
          await this.loadHistoryAndBackground()
          this.loading = false
          await this.loadRecommend() // 获取推荐信息
        }
        if (data.messageCommand === 'WAIT4PARTNER' && !this.finished) {
          this.message = 'Matching you with a chat partner'
          this.partner = ''
          this.role = ''
          this.loading = true
        }
        if (data.messageCommand === 'SENDMESSAGE' && !this.finished) {
          console.log('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
          console.log(data)
          console.log('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
          console.log(dataList)
          if (this.role === 'cus') {
            this.raw_info = data.content.split('iuiuiuiuiuiui')
            data.content = this.raw_info[0]
            console.log(this.raw_info)
            let tmp = this.raw_info[1].toString()
            this.raw_info = tmp.split('|*&*|')
            this.raw_info.pop()
            console.log(this.raw_info)
            for (let i = 0; i < this.raw_info.length; i++) {
              let tmp2 = this.raw_info[i].split('imidimid')
              let tmp3 = {name: tmp2[0], id: tmp2[1]}
              if (!this.add_info.some(item => item.name === tmp3.name && item.id === tmp3.id)) {
                this.add_info.push(tmp3)
              }
            }
            console.log('raw')
            console.log(this.raw_info)
            console.log('add')
            console.log(this.add_info)
            this.recommend_info = this.add_info
            this.$http.post('/api/saveRecommend', {
              recommend_info: this.recommend_info,
              username: this.username,
              conversationId: this.conversationId
            }).then((response) => {
              // this.$emit('completed')
            }).catch((e) => {
              this.submitting = false
              this.$Message.error('The product failed to save the database!')
            })
          }
          console.log(this.recommend_info)
          if (data.content) {
            this.$refs.chatUI.addMessage({message: data.content, time: new Date()}, 'sys')
          }
          if (this.role === 'sys') {
            this.loading = true
            this.disabled = true
            this.message = 'Loading states'
            await this.loadCurrentState()
          }
          this.loading = false
          this.disabled = false
        }
        if (data.messageCommand === 'WAIT4MESSAGE' && !this.finished) {
          if (data.content) {
            this.$refs.chatUI.addMessage({message: data.content, time: new Date()}, 'sys')
          }
          this.loading = false
          this.disabled = true
        }
        if (data.messageCommand === 'ERROR' && !this.finished) {
          if (data.conversationId >= 0) {
            this.$Notice.error(
              {
                title: 'Error',
                desc: data.content
              }
            )
          } else {
            this.loading = true
            this.$Modal.warning({
              title: 'Notice',
              content: data.content
            })
          }
        }
        if (data.messageCommand === 'STOP' && !this.finished) {
          this.$Notice.error(
            {
              title: 'Error',
              desc: 'Your partner is offline!'
            }
          )
          this.message = 'Waiting for your partner reconnect to server'
          this.loading = true
        }
        if (data.messageCommand === 'HURRYUP' && !this.finished) {
          this.$Notice.warning(
            {
              title: 'Notice',
              desc: 'Hurry up please!'
            }
          )
        }
        if (data.messageCommand === 'FINISH') {
          this.finished = true
          this.$Notice.warning(
            {
              title: 'Notice',
              desc: 'The conversationclient is finished, please rate it before leave.'
            }
          )
          this.loading = false
          this.disabled = true
        }
      })
    },
    websocketsend (data) {
      this.$socket.send(JSON.stringify({
        from: this.username,
        to: this.partner,
        type: this.role === 'sys' ? 'SYS2CUS' : 'CUS2SYS',
        ...data
      }))
    },
    hurryup () {
      this.$socket.send(JSON.stringify({
        content: 'hurry up!',
        from: this.username,
        to: this.partner,
        type: 'HURRYUP'
      }))
      this.hurryupLeftSeconds = maxHurryupLeftSeconds
      let interval = setInterval(() => {
        this.hurryupLeftSeconds--
        if (this.hurryupLeftSeconds === 0) {
          clearInterval(interval)
        }
      }, 1000)
    },
    websocketclose (e) {
      // 关闭链接时触发
      var code = e.code//  状态码表 https://developer.mozilla.org/zh-CN/docs/Web/API/CloseEvent
      var reason = e.reason
      var wasClean = e.wasClean
      console.log(code, reason, wasClean)
    },
    loadUser () {
      this.username = util.parseUrl(location.href)._user
    },
    loadHistoryAndBackground () {
      const params = {
        sysName: this.role === 'sys' ? this.username : this.partner,
        cusName: this.role === 'cus' ? this.username : this.partner,
        conversationId: this.conversationId
      }
      return this.$http.get('/api/loadHistoryAndBackground', {params}).then((response) => {
        // todo 处理history
        if (response.data.history) {
          let history = []
          response.data.history.map(value => {
            history.unshift({
              message: value.content,
              time: value.updateTime,
              type: (value.type === 'CUS2SYS' && this.role === 'cus') || (value.type === 'SYS2CUS' && this.role === 'sys') ? 'user' : 'sys'
            })
          })
          this.$refs.chatUI.addHistoy(history)
        }
        if (response.data.background) {
          this.$refs.chatUI.addMessage({
            message: '<strong>Your Role: </strong>' + (this.role === 'sys' ? 'System. ' : 'Normal user. ') + '<br/>' +
              (this.role === 'cus' ? '<strong>Background: </strong>' + response.data.background + '<br/>' : '') + '<strong>Now chat starts!</strong>' + '<br/>' + (this.role === 'cus' ? 'Note: the system response may be a bit slow since multiple steps need to be taken before sending you a response, please be patient.' : ''),
            time: new Date()
          }, 'other')
          if (this.role === 'cus') {
            this.background = response.data.background
          } else {
            this.background = 'Please try to help your partner get his/her target information.'
          }
        }
      })
    },
    loadCurrentState () {
      return this.$http.get('/api/loadCurrentState').then((response) => {
        // this.currentState = response.data
        this.currentState = ''
      })
    },
    saveStatesBackup (newStates) {
      this.statesBackupList = newStates
      console.log('NEW STATES SAVED: ' + newStates)
    },
    searchWithFilters (newFilters) {
      console.log('Searching with filters: ' + newFilters)
      console.log('Searching for the query:  ' + this.statesBackupList)
      if (this.statesBackupList.length === 0 || newFilters.length === 0) {
        return
      }
      // debugger
      let that = this
      let url = 'http://8.219.97.40:9195?query=' + encodeURIComponent(this.statesBackupList.join(' ')) + '&refinements=' + encodeURIComponent(newFilters.join(','))
      axios.get(
        url, { timeout: 300000 }
      ).then((json) => {
        if (json.data.ref === 'error') {
          this.$Notice.warning({
            title: 'Warning',
            desc: 'There are too many restrictions, please reselect'
          })
          return
        }
        // debugger
        this.$http.post('/api/saveSearchResults', {
          query: this.statesBackupList.join(' '),
          filter: newFilters.join(' '),
          conversationId: this.conversationId,
          data: json.data
        })
        console.log('Search successful.')
        this.$Notice.success({
          title: 'Success!',
          desc: 'Search successful.'
        })
        this.searchResultsBackup = json.data
        that.bus.$emit('loadMore', json.data)
      }).catch((error) => {
        console.log(error)
        this.$Notice.error({
          title: 'Error',
          desc: 'Please refresh the page or change the query to search again!'
        })
        let emptyData = {
          Answer: [],
          Suggest: [],
          Filters: [],
          Aspects: []
        }
        this.searchResultsBackup = emptyData
        that.bus.$emit('loadMore', emptyData)
      })
    },
    searchData (states, callback) {
      if (states.length === 0) {
        return
      }
      console.log('Searching for ' + states)
      this.$Notice.info({
        title: 'Searching',
        desc: 'Searching, please wait'
      })
      this.searching = true
      this.searchPanelLoading = false
      let url = 'http://8.218.97.40:9195?query=' + encodeURIComponent(states.join(' '))
      axios.get(
        url, { timeout: 300000 }
      ).then((json) => {
        if (json.data.key === 'error') {
          this.$Notice.warning({
            title: 'Warning',
            desc: 'The query is too broad, please choose a more accurate query'
          })
          let emptyData = {
            Answer: [],
            Suggest: [],
            Filters: [],
            Aspects: []
          }
          this.searchResultsBackup = emptyData
          callback(emptyData)
          return
        }
        this.$http.post('/api/saveSearchResults', {
          query: states.join(' '),
          filter: '',
          conversationId: this.conversationId,
          data: json.data
        })
        console.log('Search successful.')
        this.$Notice.success({
          title: 'Success!',
          desc: 'Search successful.'
        })
        this.searchResultsBackup = json.data
        callback(json.data)
      }).catch((error) => {
        console.log(error)
        this.$Notice.error({
          title: 'Error',
          desc: 'Please refresh the page or change the query to search again!'
        })
        let emptyData = {
          Answer: [],
          Suggest: [],
          Filters: [],
          Aspects: []
        }
        this.searchResultsBackup = emptyData
        callback(emptyData)
      })
    },
    loadAction () {
      return this.$http.get('/api/loadActions').then((response) => {
        this.actions = response.data
      })
    },
    sendData (data, callback) {
      console.log(data.sendAnother)
      this.loading = true
      this.message = 'Sending message'
      let msg = data.response
      console.log('response')
      console.log(data.response)
      console.log(this.selectedResultsBackup)
      for (let i = 0; i < this.selectedResultsBackup.length; i++) {
        let item = this.selectedResultsBackup[i]
        if (item.hasOwnProperty('link')) {
          msg += `<div style="display: flex; align-items: center;"><img src="${item.image}" alt="${item.title}" height="50px" width="50px" style="margin-right: 10px;"> <a style="color: yellowgreen" target="_blank" href="${item.link}">${item.title}</a></div>`
        }
        if (i < this.selectedResultsBackup.length - 1) {
          msg += '<br>' // Add an empty line between results
        }
      }
      let msg2 = msg
      if (this.role === 'sys') {
        msg += 'iuiuiuiuiuiui'
        for (let i = 0; i < this.recommend_info.length; i++) {
          msg += this.recommend_info[i]
          const id = md5(name).toString()
          msg += 'imidimid' + id
          msg += '|*&*|'
        }
      }
      data.response = msg
      if (this.role === 'sys') {
        msg = msg2
      }
      console.log('data:')
      console.log(data)
      this.websocketsend(data)
      this.$refs.chatUI.addMessage({message: msg2, time: new Date()}, 'user')
      callback()
    },
    saveChangeBackup (checked) {
      this.selectedResultsBackup = checked
    }
  }
}
</script>
